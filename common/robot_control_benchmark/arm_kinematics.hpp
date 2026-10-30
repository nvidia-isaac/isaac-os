/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// 7-DoF FK and a fixed-iteration damped-least-squares IK step (KUKA LBR iiwa 7
// R800 DH). Warm-started from the previous solution; per-tick cost is
// deterministic. Hand-rolled linear algebra, no Eigen.
//
// The model is templated on its scalar type and callable from both host and device, so
// the CPU and GPU benchmark backends run one implementation rather than two. ArmModel
// keeps the double instantiation every existing caller expects; the GPU solve
// instantiates it as float. Keeping a single definition removes the need for a separate
// device-side mirror and an equivalence test that polices the two for drift.
//
// Two constraints on anything added here, both load-bearing:
//
//   * Every function reachable from the GPU solve needs RCB_HD, and the annotation has to
//     vanish off-nvcc: most translation units including this header are plain host C++
//     compiled without nvcc, and would not recognize the attributes.
//   * Nothing on the device path may call a constexpr host function. std::array's
//     subscript and std::clamp are exactly that, so device code would need nvcc's
//     --expt-relaxed-constexpr to call them. Mat4T carries its own subscript and
//     kin_clamp replaces std::clamp specifically so no CUDA target here needs that flag.
//     std::array is still fine in the host-only parts, such as home().

#include <math.h>

#include <array>
#include <cmath>
#include <numbers>

#include "common/robot_control_benchmark/payloads.hpp"

#if defined(__CUDACC__)
#define RCB_HD __host__ __device__
#else
#define RCB_HD
#endif

namespace isaac::os::benchmark {

namespace detail {

/// Computes a host/device float sine without widening the input.
RCB_HD inline float kin_sin(float v) { return sinf(v); }
/// Computes a host/device double sine.
RCB_HD inline double kin_sin(double v) { return sin(v); }
/// Computes a host/device float cosine without widening the input.
RCB_HD inline float kin_cos(float v) { return cosf(v); }
/// Computes a host/device double cosine.
RCB_HD inline double kin_cos(double v) { return cos(v); }
/// Computes a host/device float square root without widening the input.
RCB_HD inline float kin_sqrt(float v) { return sqrtf(v); }
/// Computes a host/device double square root.
RCB_HD inline double kin_sqrt(double v) { return sqrt(v); }

/// Clamps a scalar without calling the host-only `std::clamp` from device code.
template <typename T>
RCB_HD inline T kin_clamp(T value, T low, T high) {
  if (value < low) {
    return low;
  }
  if (value > high) {
    return high;
  }
  return value;
}

}  // namespace detail

/// Host/device 4x4 homogeneous transform stored in row-major order.
template <typename T>
struct Mat4T {
  T m[16];  // NOSONAR: fixed storage must remain callable from CUDA device code.

  /// Returns a mutable row-major matrix element.
  RCB_HD T& operator[](int index) { return m[index]; }
  /// Returns a row-major matrix element.
  RCB_HD const T& operator[](int index) const { return m[index]; }
};

using Mat4 = Mat4T<double>;

/// Returns a host/device identity transform for the requested scalar type.
template <typename T>
RCB_HD Mat4T<T> mat4_identity_of() {
  return Mat4T<T>{{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}};
}

/// Returns a double-precision identity transform for host callers.
inline Mat4 mat4_identity() { return mat4_identity_of<double>(); }

/// Multiplies two host/device homogeneous transforms.
template <typename T>
RCB_HD Mat4T<T> mat4_mul(const Mat4T<T>& a, const Mat4T<T>& b) {
  Mat4T<T> r{};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      T s = 0.0;
      for (int k = 0; k < 4; ++k) {
        s += a[i * 4 + k] * b[k * 4 + j];
      }
      r[i * 4 + j] = s;
    }
  }
  return r;
}

/// Builds a standard distal Denavit-Hartenberg link transform.
template <typename T>
RCB_HD Mat4T<T> dh_transform(T theta, T d, T a, T alpha) {
  const T ct = detail::kin_cos(theta);
  const T st = detail::kin_sin(theta);
  const T ca = detail::kin_cos(alpha);
  const T sa = detail::kin_sin(alpha);
  return Mat4T<T>{
      {ct, -st * ca, st * sa, a * ct, st, ct * ca, -ct * sa, a * st, 0, sa, ca, d, 0, 0, 0, 1}};
}

/// Builds a row-major rotation matrix as `Rz(yaw) * Ry(pitch) * Rx(roll)`.
inline std::array<double, 9> rpy_to_rotation(double roll, double pitch, double yaw) {
  const double cr = std::cos(roll), sr = std::sin(roll);
  const double cp = std::cos(pitch), sp = std::sin(pitch);
  const double cy = std::cos(yaw), sy = std::sin(yaw);
  return std::array<double, 9>{cy * cp,
                               cy * sp * sr - sy * cr,
                               cy * sp * cr + sy * sr,
                               sy * cp,
                               sy * sp * sr + cy * cr,
                               sy * sp * cr - cy * sr,
                               -sp,
                               cp * sr,
                               cp * cr};
}

/// KUKA LBR iiwa 7 R800 forward-kinematics and fixed-step DLS IK model.
template <typename T>
class ArmModelT {
 public:
  static constexpr int N = kNumJoints;  // 7

  /// Computes the end-effector transform and joint frames used by the Jacobian.
  RCB_HD void forward(const T* q, Mat4T<T>& ee, Mat4T<T> (&frames)[N + 1]) const {
    Mat4T<T> cumulative = mat4_identity_of<T>();
    frames[0] = cumulative;
    for (int i = 0; i < N; ++i) {
      cumulative = mat4_mul<T>(cumulative, dh_transform<T>(q[i], d_[i], a_[i], alpha_[i]));
      frames[i + 1] = cumulative;
    }
    ee = cumulative;
  }

  /// Computes the row-major geometric Jacobian for the supplied joint frames.
  RCB_HD void jacobian(const Mat4T<T> (&frames)[N + 1], const Mat4T<T>& ee, T* J) const {
    const T pe[3] = {ee[3], ee[7], ee[11]};  // NOSONAR: fixed CUDA device storage.
    for (int i = 0; i < N; ++i) {
      const Mat4T<T>& f = frames[i];
      const T z[3] = {f[2], f[6], f[10]};  // NOSONAR: fixed CUDA device storage.
      const T p[3] = {f[3], f[7], f[11]};  // NOSONAR: fixed CUDA device storage.
      const T r[3] = {                     // NOSONAR: fixed CUDA device storage.
                      pe[0] - p[0], pe[1] - p[1], pe[2] - p[2]};
      // linear part = z x r
      J[0 * N + i] = z[1] * r[2] - z[2] * r[1];
      J[1 * N + i] = z[2] * r[0] - z[0] * r[2];
      J[2 * N + i] = z[0] * r[1] - z[1] * r[0];
      // angular part = z
      J[3 * N + i] = z[0];
      J[4 * N + i] = z[1];
      J[5 * N + i] = z[2];
    }
  }

  /// Runs fixed-count DLS IK iterations in place from the supplied warm start.
  RCB_HD void ik_step(T* q, const T target_p[3], const T target_R[9],
                      int iterations = kIkIterations, T lambda = static_cast<T>(0.05),
                      T max_dq = static_cast<T>(0.1)) const {
    for (int it = 0; it < iterations; ++it) {
      Mat4T<T> ee{};
      // forward() overwrites every element before jacobian() reads it.
      Mat4T<T> frames[N + 1];  // NOSONAR: fixed CUDA device storage.
      forward(q, ee, frames);

      // Pose error e[6] = [position error; orientation error].
      T e[6];  // NOSONAR: fixed CUDA device storage.
      e[0] = target_p[0] - ee[3];
      e[1] = target_p[1] - ee[7];
      e[2] = target_p[2] - ee[11];
      orientation_error(ee, target_R, &e[3]);

      T J[6 * N];  // NOSONAR: fixed CUDA device storage.
      jacobian(frames, ee, J);

      T A[6][6];  // NOSONAR: fixed CUDA device storage.
      form_damped_system(J, lambda, A);
      T y[6];  // NOSONAR: fixed CUDA device storage.
      if (!chol_solve6(A, e, y)) {
        return;  // ill-conditioned; keep last good q
      }
      apply_joint_update(q, J, y, max_dq);
    }
  }

  /// Returns the default home-like warm start.
  static std::array<T, N> home() { return std::array<T, N>{0.0, 0.4, 0.0, -1.2, 0.0, 1.0, 0.0}; }

 private:
  static constexpr int kIkIterations = 4;
  static constexpr T kHalfPi = std::numbers::pi_v<T> / static_cast<T>(2);

  T d_[N] = {0.34, 0.0, 0.40, 0.0, 0.40, 0.0, 0.126};  // NOSONAR: CUDA storage.
  T a_[N] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};       // NOSONAR: CUDA storage.
  T alpha_[N] = {                                      // NOSONAR: fixed CUDA device storage.
      -kHalfPi, kHalfPi, kHalfPi, -kHalfPi, -kHalfPi, kHalfPi, 0.0};

  /// Computes the base-frame orientation error between current and desired rotations.
  RCB_HD static void orientation_error(const Mat4T<T>& ee, const T Rd[9], T* eo) {
    eo[0] = eo[1] = eo[2] = 0.0;
    for (int c = 0; c < 3; ++c) {
      const T rc[3] = {// NOSONAR: fixed CUDA device storage.
                       ee[0 * 4 + c], ee[1 * 4 + c], ee[2 * 4 + c]};
      const T rd[3] = {// NOSONAR: fixed CUDA device storage.
                       Rd[0 * 3 + c], Rd[1 * 3 + c], Rd[2 * 3 + c]};
      eo[0] += static_cast<T>(0.5) * (rc[1] * rd[2] - rc[2] * rd[1]);
      eo[1] += static_cast<T>(0.5) * (rc[2] * rd[0] - rc[0] * rd[2]);
      eo[2] += static_cast<T>(0.5) * (rc[0] * rd[1] - rc[1] * rd[0]);
    }
  }

  /// Forms the damped six-dimensional normal system for the Jacobian.
  RCB_HD static void form_damped_system(const T* jacobian, T lambda, T (&system)[6][6]) {
    for (int row = 0; row < 6; ++row) {
      for (int column = 0; column < 6; ++column) {
        T sum = 0.0;
        for (int joint = 0; joint < N; ++joint) {
          sum += jacobian[row * N + joint] * jacobian[column * N + joint];
        }
        system[row][column] = sum + (row == column ? lambda * lambda : static_cast<T>(0));
      }
    }
  }

  /// Applies the clamped joint update produced by the damped solve.
  RCB_HD static void apply_joint_update(T* q, const T* jacobian, const T* solution, T max_delta) {
    for (int joint = 0; joint < N; ++joint) {
      T delta = 0.0;
      for (int row = 0; row < 6; ++row) {
        delta += jacobian[row * N + joint] * solution[row];
      }
      q[joint] += detail::kin_clamp(delta, -max_delta, max_delta);
    }
  }

  /// Returns one residual while constructing the Cholesky factor.
  RCB_HD static T cholesky_residual(const T matrix[6][6], const T lower[6][6], int row,
                                    int column) {
    T result = matrix[row][column];
    for (int index = 0; index < column; ++index) {
      result -= lower[row][index] * lower[column][index];
    }
    return result;
  }

  /// Solves a 6x6 SPD system by Cholesky and returns false when it is not positive definite.
  RCB_HD static bool chol_solve6(const T A[6][6], const T b[6], T x[6]) {
    T L[6][6] = {};  // NOSONAR: fixed CUDA device storage.
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j <= i; ++j) {
        const T sum = cholesky_residual(A, L, i, j);
        if (i == j && sum <= static_cast<T>(0)) {
          return false;
        }
        L[i][j] = i == j ? detail::kin_sqrt(sum) : sum / L[j][j];
      }
    }
    T y[6];  // NOSONAR: fixed CUDA device storage.
    for (int i = 0; i < 6; ++i) {
      T sum = b[i];
      for (int k = 0; k < i; ++k) {
        sum -= L[i][k] * y[k];
      }
      y[i] = sum / L[i][i];
    }
    for (int i = 5; i >= 0; --i) {
      T sum = y[i];
      for (int k = i + 1; k < 6; ++k) {
        sum -= L[k][i] * x[k];
      }
      x[i] = sum / L[i][i];
    }
    return true;
  }
};

using ArmModel = ArmModelT<double>;

/// Returns whether forward kinematics for the measured joints is finite.
[[nodiscard]] inline bool measured_joints_fk_finite(const double* q) {
  ArmModel model;
  Mat4 ee{};
  Mat4 frames[ArmModel::N + 1]{};  // NOSONAR: matches the CUDA-compatible model API.
  model.forward(q, ee, frames);
  return std::isfinite(ee[3]) && std::isfinite(ee[7]) && std::isfinite(ee[11]);
}

}  // namespace isaac::os::benchmark
