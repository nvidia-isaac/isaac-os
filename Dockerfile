# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

############################################################
# Versions
############################################################
ARG CUDA_MAJOR=13
ARG HOLOSCAN_VERSION=5.0.0

############################################################
# Base image
############################################################
ARG HOLOSCAN_IMAGE_REPO=nvcr.io/nvidia/clara-holoscan/holoscan
FROM ${HOLOSCAN_IMAGE_REPO}:v${HOLOSCAN_VERSION}-cuda13 AS cu13-base
FROM cu${CUDA_MAJOR}-base AS base

# Set bash as the default shell
SHELL ["/bin/bash", "-c"]

# Common variables
ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ARG TARGETARCH
ARG CUDA_MAJOR

# Conditionally configure apt caching behavior
ARG ENABLE_APT_CACHING=true
RUN if [ "${ENABLE_APT_CACHING}" = "true" ]; then \
        echo "APT Caching enabled: Disabling docker-clean and enabling Keep-Downloaded-Packages."; \
        DOCKER_CLEAN_CONF="/etc/apt/apt.conf.d/docker-clean"; \
        if [ -f "${DOCKER_CLEAN_CONF}" ]; then \
            mv "${DOCKER_CLEAN_CONF}" "${DOCKER_CLEAN_CONF}.disabled"; \
        fi; \
        echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/99-keep-archives; \
    else \
        echo "APT Caching disabled: Adding extra Post-Invoke cleanup hook."; \
        echo 'DPkg::Post-Invoke "rm -f /var/cache/apt/archives/*.deb /var/cache/apt/archives/partial/*.deb /var/cache/apt/*.bin /var/lib/apt/lists/* || true";' > /etc/apt/apt.conf.d/99-custom-cleanup; \
    fi
ARG MOUNT_CACHE_PREFIX="isaac-${TARGETARCH}"
ARG APT_CACHE_ID="${MOUNT_CACHE_PREFIX}-apt-cache"
ARG APT_LIB_ID="${MOUNT_CACHE_PREFIX}-apt-lib"
ARG PIP_CACHE_ID="${MOUNT_CACHE_PREFIX}-pip-cache"
ARG UV_CACHE_ID="${MOUNT_CACHE_PREFIX}-uv-cache"
ARG CARGO_REGISTRY_CACHE_ID="${MOUNT_CACHE_PREFIX}-cargo-registry"
ARG CARGO_GIT_CACHE_ID="${MOUNT_CACHE_PREFIX}-cargo-git"
ARG SCCACHE_CACHE_ID="${MOUNT_CACHE_PREFIX}-sccache"

############################################################
# Python-base
############################################################
FROM base AS python-base

ENV PIP_BREAK_SYSTEM_PACKAGES=1
ARG PYTHON_VERSION
ARG PYTHON_BIN=python${PYTHON_VERSION}
ARG GET_PIP_COMMIT=f6f644156f23dfe9acc06e7b9ca75eee311f2e37
ARG GET_PIP_SHA256=fb24e693bab954209a063d90953621412ccad4a500905a726286e038f508ddf6
# This is the chain-of-attack safe alternative to https://bootstrap.pypa.io/get-pip.py
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=${APT_CACHE_ID} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=${APT_LIB_ID} \
    apt-get update \
    && if [ -n "${PYTHON_VERSION}" ]; then \
        apt-get install --no-install-recommends -y \
            software-properties-common \
        && add-apt-repository ppa:deadsnakes/ppa \
        && apt-get update \
        && apt-get install --no-install-recommends -y \
            ${PYTHON_BIN} \
            ${PYTHON_BIN}-dev \
        && apt purge -y \
            python3-pip \
            software-properties-common \
        && apt-get autoremove --purge -y \
        && curl --proto '=https' --tlsv1.2 -fsSL -o /tmp/get-pip.py \
            "https://raw.githubusercontent.com/pypa/get-pip/${GET_PIP_COMMIT}/public/get-pip.py" \
        && printf '%s  %s\n' "${GET_PIP_SHA256}" /tmp/get-pip.py \
            | sha256sum --check --strict - \
        && "${PYTHON_BIN}" /tmp/get-pip.py \
        && rm /tmp/get-pip.py \
        && update-alternatives --install /usr/bin/python3 python3 /usr/bin/${PYTHON_BIN} 1; \
    else \
        apt-get install --no-install-recommends -y \
            python3-dev \
            python3-pip; \
    fi

RUN python3 --version | grep "${PYTHON_VERSION}" && pip --version | grep "${PYTHON_VERSION}"

############################################################
# Build tools
############################################################
FROM python-base AS build-tools

# Install ld.mold
ARG MOLD_VERSION=2.40.4
ARG MOLD_X86_64_SHA256=4c999e19ffa31afa5aa429c679b665d5e2ca5a6b6832ad4b79668e8dcf3d8ec1
ARG MOLD_AARCH64_SHA256=c799b9ccae8728793da2186718fbe53b76400a9da396184fac0c64aa3298ec37
RUN arch="$(uname -m)" \
    && case "${arch}" in \
        x86_64) mold_sha256="${MOLD_X86_64_SHA256}" ;; \
        aarch64) mold_sha256="${MOLD_AARCH64_SHA256}" ;; \
        *) echo "Unsupported mold architecture: ${arch}" >&2; exit 1 ;; \
    esac \
    && curl --proto '=https' --tlsv1.2 -fSL -o /tmp/mold.tar.gz \
        "https://github.com/rui314/mold/releases/download/v${MOLD_VERSION}/mold-${MOLD_VERSION}-${arch}-linux.tar.gz" \
    && printf '%s  %s\n' "${mold_sha256}" /tmp/mold.tar.gz \
        | sha256sum --check --strict - \
    && tar -xzf /tmp/mold.tar.gz -C /usr/local \
        --strip-components=1 --no-same-owner --no-same-permissions \
    && rm /tmp/mold.tar.gz

# Install CPM.cmake (CMake Package Manager)
ARG CPM_VERSION=0.42.1
ARG CPM_SHA256=f3a6dcc6a04ce9e7f51a127307fa4f699fb2bade357a8eb4c5b45df76e1dc6a5
ENV CPM_CMAKE_FILE=/usr/local/share/cmake/Modules/CPM.cmake
RUN mkdir -p "$(dirname "${CPM_CMAKE_FILE}")" \
    && curl --proto '=https' --tlsv1.2 -fSL -o "${CPM_CMAKE_FILE}" \
        "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_VERSION}/CPM.cmake" \
    && printf '%s  %s\n' "${CPM_SHA256}" "${CPM_CMAKE_FILE}" \
        | sha256sum --check --strict -

ARG CMAKE_BUILD_TYPE=Release

############################################################
# Video Codec SDK
############################################################
FROM build-tools AS video-codec-downloader

# NVIDIA publishes this Debian package for ARM64 Jetson, so apt cannot install it on x64.
# Its headers and the encoder/decoder helper sources are portable C/C++ compiled for the
# build architecture, so we extract the same files for x64 and ARM64. Keep the license
# files, but exclude ARM64 stub libraries and sample applications we do not use.
ARG VIDEO_CODEC_VERSION=13.0.37
ARG VIDEO_CODEC_DEB_VERSION=${VIDEO_CODEC_VERSION}-2
ARG VIDEO_CODEC_SHA256=63047d99b1b65d472b48dac2c821d26a32c102a42c2e20c8df499b1194216c36
ADD --checksum=sha256:${VIDEO_CODEC_SHA256} \
    https://repo.download.nvidia.com/jetson/common/pool/main/n/nvidia-video-codec-sdk/nvidia-video-codec-sdk_${VIDEO_CODEC_DEB_VERSION}_arm64.deb \
    /tmp/video-codec.deb
RUN dpkg-deb --extract /tmp/video-codec.deb /tmp/video-codec \
    && mkdir -p /opt/nvidia/video-codec-sdk/Samples/NvCodec \
    && cp -a "/tmp/video-codec/opt/nvidia/video-codec-sdk/${VIDEO_CODEC_VERSION}/"{Interface,LicenseAgreement.pdf,NOTICES.txt} \
        /opt/nvidia/video-codec-sdk/ \
    && cp -a "/tmp/video-codec/opt/nvidia/video-codec-sdk/${VIDEO_CODEC_VERSION}/Samples/NvCodec/"{NvDecoder,NvEncoder} \
        /opt/nvidia/video-codec-sdk/Samples/NvCodec/ \
    && cp -a "/tmp/video-codec/opt/nvidia/video-codec-sdk/${VIDEO_CODEC_VERSION}/Samples/Utils" \
        /opt/nvidia/video-codec-sdk/Samples/ \
    && cp /tmp/video-codec/usr/share/doc/nvidia-video-codec-sdk/copyright \
        /opt/nvidia/video-codec-sdk/copyright

############################################################
# sccache
############################################################
FROM build-tools AS sccache-downloader

WORKDIR /opt/sccache

ENV SCCACHE_VERSION=v0.14.0-rapids.13
ENV SCCACHE_BASE_URL=https://github.com/rapidsai/sccache/releases/download
ARG SCCACHE_X86_64_SHA256=7e50574d0cd9e636b1c357490543a061fae2b088c75e1231e6645d75e53a1ce6
ARG SCCACHE_AARCH64_SHA256=71f34ed1bc687a47aa1c895ca316430cc51404d01c924d90cf5c90f6955ac059

RUN arch="$(uname -m)" \
    && case "${arch}" in \
        x86_64) sccache_sha256="${SCCACHE_X86_64_SHA256}" ;; \
        aarch64) sccache_sha256="${SCCACHE_AARCH64_SHA256}" ;; \
        *) echo "Unsupported sccache architecture: ${arch}" >&2; exit 1 ;; \
    esac \
    && curl --proto '=https' --tlsv1.2 -fSL -o sccache.tar.gz \
        "${SCCACHE_BASE_URL}/${SCCACHE_VERSION}/sccache-${SCCACHE_VERSION}-${arch}-unknown-linux-musl.tar.gz" \
    && printf '%s  %s\n' "${sccache_sha256}" sccache.tar.gz \
        | sha256sum --check --strict - \
    && tar -xzf sccache.tar.gz -C /opt/sccache --strip-components=1 \
        "sccache-${SCCACHE_VERSION}-${arch}-unknown-linux-musl/sccache" \
    && rm sccache.tar.gz

############################################################
# uv
############################################################
FROM build-tools AS uv-downloader

WORKDIR /opt/uv

ARG UV_X86_64_SHA256=b4dfaef47d491a7296981f8374a4595f55dbf84e8937c8ecd2983574d8bb3da6
ARG UV_AARCH64_SHA256=5231be65f496304623895dacdbf1de8504fec90303684bdf05805aa34414dd21

RUN arch="$(uname -m)" \
    && case "${arch}" in \
        x86_64) uv_sha256="${UV_X86_64_SHA256}" ;; \
        aarch64) uv_sha256="${UV_AARCH64_SHA256}" ;; \
        *) echo "Unsupported uv architecture: ${arch}" >&2; exit 1 ;; \
    esac \
    && curl --proto '=https' --tlsv1.2 -fSL -o uv.tar.gz \
        "https://github.com/astral-sh/uv/releases/download/0.12.24/uv-${arch}-unknown-linux-gnu.tar.gz" \
    && printf '%s  %s\n' "${uv_sha256}" uv.tar.gz \
        | sha256sum --check --strict - \
    && tar -xzf uv.tar.gz -C /opt/uv --strip-components=1 \
        "uv-${arch}-unknown-linux-gnu/uv" \
    && rm uv.tar.gz

############################################################
# ROS 2 base
############################################################
FROM build-tools AS ros-base

# Install ROS 2
# See https://docs.ros.org/en/lyrical/Installation/Ubuntu-Install-Debs.html
# On noble we use ROS 2 Lyrical, which packages.ros.org only publishes for Ubuntu 26.04, so the
# Isaac ROS Buildfarm provides the Lyrical packages, including rosidl::Buffer. The CUDA buffer
# backend that lets ROS messages carry GPU memory without a copy is built from source in
# ros-cuda-buffer-builder below: the Buildfarm only publishes it for CUDA 12, and without it a
# CUDA-backed rosidl::Buffer is silently converted to host memory in transit.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=${APT_CACHE_ID} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=${APT_LIB_ID} \
    set -e \
    && OS_CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME") \
    && if [ "$OS_CODENAME" = "jammy" ]; then \
        ROS_DISTRO="humble"; \
        ROS_BASE_VER="0.10"; \
        RCLCPP_VER="16.0"; \
        ROS_MSGS_VER="4.9"; \
        ROS_EXTRA_PKGS=(); \
    elif [ "$OS_CODENAME" = "noble" ]; then \
        ROS_DISTRO="lyrical"; \
        ROS_BASE_VER="0.13"; \
        RCLCPP_VER="32.0"; \
        ROS_MSGS_VER="5.9"; \
        ROS_EXTRA_PKGS=( \
            "ros-${ROS_DISTRO}-rosidl-buffer=5.2.*" \
        ); \
        ISAAC_ROS_KEYRING="/usr/share/keyrings/nvidia-isaac-ros.gpg"; \
        curl --proto '=https' --tlsv1.2 -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key \
            | gpg --batch --dearmor -o "$ISAAC_ROS_KEYRING"; \
        echo "deb [arch=$TARGETARCH signed-by=$ISAAC_ROS_KEYRING] https://isaac.download.nvidia.com/isaac-ros/ubuntu/main noble main" \
            > /etc/apt/sources.list.d/isaac-ros-buildfarm.list; \
    else \
        echo "ERROR: Unsupported OS for ROS 2: ${OS_CODENAME}" >&2; \
        exit 1; \
    fi \
    && ROS2_KEYRING="/usr/share/keyrings/ros-archive-keyring.gpg" \
    && curl -fsSL "https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc" \
        | gpg --batch --dearmor -o "$ROS2_KEYRING" \
    && echo "deb [arch=$TARGETARCH signed-by=$ROS2_KEYRING] http://packages.ros.org/ros2/ubuntu $OS_CODENAME main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        "python3-colcon-common-extensions=0.3.*" \
        "python3-rosdep=0.27.*" \
        "ros-${ROS_DISTRO}-ros-base=${ROS_BASE_VER}.*" \
        "ros-${ROS_DISTRO}-rclcpp=${RCLCPP_VER}.*" \
        "ros-${ROS_DISTRO}-sensor-msgs=${ROS_MSGS_VER}.*" \
        "ros-${ROS_DISTRO}-std-msgs=${ROS_MSGS_VER}.*" \
        "${ROS_EXTRA_PKGS[@]}" \
    && ROS_SETUP_PATH="/opt/ros/setup.bash" \
    && echo "export ROS_DISTRO=${ROS_DISTRO}" >> ${ROS_SETUP_PATH} \
    && echo "export ROS_VERSION=2" >> ${ROS_SETUP_PATH} \
    && echo "export PYTHONPATH=/opt/ros/${ROS_DISTRO}/lib/python3/dist-packages:\${PYTHONPATH:-}" >> ${ROS_SETUP_PATH} \
    && echo "set +u; source /opt/ros/${ROS_DISTRO}/setup.bash; set -u;" >> ${ROS_SETUP_PATH} \
    && chmod +x ${ROS_SETUP_PATH}

############################################################
# ROS CUDA buffer backend builder
############################################################
FROM ros-base AS ros-cuda-buffer-builder

# Build the CUDA backend for rosidl::Buffer (cuda_buffer, cuda_buffer_backend and
# cuda_buffer_backend_msgs) from the upstream ROS repository against this image's CUDA toolkit
# into an overlay prefix. Pinned by commit; the commit is the upstream 0.1.2 tag.
# The three packages are built together from one commit so their headers, message types and
# plugin stay consistent. Only noble has rosidl::Buffer, so other releases get an empty overlay.
ARG ROSIDL_BUFFER_BACKENDS_REPO=https://github.com/ros2/rosidl_buffer_backends.git
ARG ROSIDL_BUFFER_BACKENDS_REF=7e723061d03b347bedea69009b003f33e9b53314
ARG ROS_OVERLAY_DIR=/opt/isaac_os/ros_overlay
RUN set -e \
    && OS_CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME") \
    && mkdir -p "${ROS_OVERLAY_DIR}" \
    && if [ "$OS_CODENAME" = "noble" ]; then \
        src=/tmp/rosidl_buffer_backends \
        && git init -q "${src}" \
        && git -C "${src}" fetch -q --depth 1 "${ROSIDL_BUFFER_BACKENDS_REPO}" "${ROSIDL_BUFFER_BACKENDS_REF}" \
        && git -C "${src}" checkout -q FETCH_HEAD \
        && source /opt/ros/setup.bash \
        && colcon --log-base /tmp/colcon_log build \
            --base-paths "${src}/cuda_buffer_backend" \
            --build-base /tmp/colcon_build \
            --install-base "${ROS_OVERLAY_DIR}" \
            --merge-install \
            --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
        && test -f "${ROS_OVERLAY_DIR}/share/ament_index/resource_index/rosidl_buffer_backend__pluginlib__plugin/cuda_buffer_backend" \
        && { set +u; source "${ROS_OVERLAY_DIR}/local_setup.bash"; set -u; } \
        && for lib in libcuda_buffer.so libcuda_buffer_backend.so; do \
            unresolved="$(ldd "${ROS_OVERLAY_DIR}/lib/${lib}" | grep 'not found' | grep -v 'libcuda.so.1' || true)" \
            && if [ -n "${unresolved}" ]; then \
                echo "ERROR: ${lib} has unresolved libraries: ${unresolved}" >&2; exit 1; \
            fi \
            && if ldd "${ROS_OVERLAY_DIR}/lib/${lib}" | grep -q 'libcudart.so.12'; then \
                echo "ERROR: ${lib} links the CUDA 12 runtime" >&2; exit 1; \
            fi; \
        done \
        && rm -rf "${src}" /tmp/colcon_build /tmp/colcon_log; \
    fi

############################################################
# Build image
############################################################
FROM ros-base AS build

# APT dependencies:
#  libevdev-dev: gamepad/evdev reading + name=code resolution (gamepad_teleop)
#  liblz4-dev: LZ4 chunk compression for MCAP logging and replay
#  nlohmann-json3-dev: JSON for Modern C++ (SO-101 calibration parsing)
#  xvfb: virtual framebuffer X11 server for headless testing
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=${APT_CACHE_ID} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=${APT_LIB_ID} \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        libevdev-dev \
        liblz4-dev \
        nlohmann-json3-dev \
        xvfb

# Layer the CUDA buffer backend over the apt-installed ROS 2. isaac-env-setup below sources the
# overlay after ROS 2. The overlay is empty on releases without it.
ARG ROS_OVERLAY_DIR=/opt/isaac_os/ros_overlay
ENV ROS_OVERLAY_DIR=${ROS_OVERLAY_DIR}
COPY --from=ros-cuda-buffer-builder ${ROS_OVERLAY_DIR} ${ROS_OVERLAY_DIR}

# Workaround: numpy 2.x moved core/ to _core/. Pre-built ROS packages reference the old path.
RUN NUMPY_DIR=$(pip show numpy | grep -oP '(?<=Location: ).+' | head -1)/numpy \
    && if [ -d "${NUMPY_DIR}/_core/include" ] && [ ! -e "${NUMPY_DIR}/core/include" ]; then \
        mkdir -p "${NUMPY_DIR}/core" \
        && ln -s "${NUMPY_DIR}/_core/include" "${NUMPY_DIR}/core/include"; \
    fi

# UCX C library + UCXX C++ wrapper prebundle.
# HSDK5 does not ship UCX/UCXX. We mirror Isaac Sim's `pip_ucx_prebundle`
# approach: install the pre-built RAPIDS wheels into /opt/ucx_prebundle, then
# our CMake references headers under lib{ucx,ucxx,rmm,rapids_logger}/include
# and shared libs under .../lib or .../lib64. Versions pinned to match Isaac
# Sim's currently shipped bundle.
# Keep HSDK5's prebundle dependencies local to its libraries.
# A global LD_LIBRARY_PATH would override SDK-provided UCX/RMM libraries.
# Patch search paths in the install step to avoid adding another image layer.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked,id=${PIP_CACHE_ID} \
    mkdir -p /opt/ucx_prebundle \
    && python3 -m pip install \
        --target=/opt/ucx_prebundle \
        --no-deps \
        --extra-index-url https://pypi.nvidia.com \
        libucx-cu12==1.19.0 \
        libucxx-cu12==0.44.0 \
        librmm-cu12==25.6.0 \
        rapids-logger==0.1.1 \
    && patchelf --add-rpath \
        '$ORIGIN/../../libucx/lib:$ORIGIN/../../librmm/lib64:$ORIGIN/../../rapids_logger/lib64' \
        /opt/ucx_prebundle/libucxx/lib64/libucxx.so \
    && patchelf --add-rpath \
        '$ORIGIN/../../rapids_logger/lib64' \
        /opt/ucx_prebundle/librmm/lib64/librmm.so

# Copy sccache
ENV sccache_ROOT=/opt/sccache
ENV SCCACHE_CACHE_SIZE=20G
COPY --from=sccache-downloader ${sccache_ROOT} /usr/local/bin

ENV NVC_SDK_ROOT=/opt/nvidia/video-codec-sdk
COPY --from=video-codec-downloader ${NVC_SDK_ROOT} ${NVC_SDK_ROOT}

# Install the locked holoscan-cli into the system Python.
# Build policy for wheel vs git pins lives in [tool.uv] in pyproject.toml.
RUN --mount=type=bind,from=uv-downloader,source=/opt/uv/uv,target=/usr/local/bin/uv \
    --mount=type=bind,source=pyproject.toml,target=/tmp/isaac-os-project/pyproject.toml \
    --mount=type=bind,source=uv.lock,target=/tmp/isaac-os-project/uv.lock \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked,id=${UV_CACHE_ID} \
    UV_PROJECT_ENVIRONMENT="$(python3 -c 'import sys; print(sys.prefix)')" \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_PYTHON_DOWNLOADS=never \
    uv sync --no-build --frozen --inexact --no-install-project --python python3 --project /tmp/isaac-os-project
ENV ISAAC_OS_CONTAINER=1

############################################################
# Entrypoint
############################################################

# Create environment setup script that bash automatically sources via BASH_ENV.
COPY --chmod=644 <<'EOF' /usr/local/bin/isaac-env-setup
# Prevent re-sourcing:
[ -n "${ISAAC_ENV_LOADED:-}" ] && return 0

# Preserve NVIDIA's CUDA compatibility initialization when overriding BASH_ENV.
source /etc/shinit_v2

# If running with a non-root user via --user, ensure HOME is set and writable.
user_id="$(id -u)"
if [ "$user_id" -ne 0 ]; then
    if [ -z "${HOME:-}" ] || [ ! -w "$HOME" ]; then
        export HOME="/tmp/home-$user_id"
        mkdir -p "$HOME"
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi

# Set up ROS 2 environment.
# See https://docs.ros.org/en/lyrical/Installation/Ubuntu-Install-Debs.html#environment-setup
source /opt/ros/setup.bash

# Layer the CUDA buffer backend over ROS 2 (the overlay is empty on releases without it).
if [ -n "${ROS_OVERLAY_DIR:-}" ] && [ -f "${ROS_OVERLAY_DIR}/local_setup.bash" ]; then
    set +u
    source "${ROS_OVERLAY_DIR}/local_setup.bash"
    set -u
fi

export ISAAC_ENV_LOADED=1
EOF

ENV BASH_ENV=/usr/local/bin/isaac-env-setup

COPY --chmod=755 <<'EOF' /usr/local/bin/isaac-entrypoint
#!/bin/bash
source /usr/local/bin/isaac-env-setup
exec "$@"
EOF

ENTRYPOINT ["/usr/local/bin/isaac-entrypoint"]
CMD ["/bin/bash"]
WORKDIR /workspace
