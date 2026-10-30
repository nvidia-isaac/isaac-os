# Security Policy: Isaac OS

NVIDIA is dedicated to the security and trust of our software products and
services, including all source code repositories managed through our
organization.

If you need to report a security issue, please use the appropriate contact
points outlined below. **Please do not report security vulnerabilities through
GitHub/GitLab issues or merge requests.**

## Reporting a Vulnerability

If you discover a potential security vulnerability, please **do not open a
public issue.**

* **Web (preferred):** [NVIDIA Vulnerability Disclosure Program](https://www.nvidia.com/en-us/security/)
* **E-Mail:** [psirt@nvidia.com](mailto:psirt@nvidia.com)
  * We encourage you to use the following PGP key for secure email
    communication: [NVIDIA public PGP Key](https://www.nvidia.com/en-us/security/pgp-key)
* **GitHub/GitLab:** Use this repository's **Security** tab (or equivalent
  private vulnerability reporting) to submit a report directly. Do not file a
  public issue or merge request for a vulnerability.

Please include the following information:

* Product/project name and version or branch (Isaac OS / this repository)
* Type of vulnerability
* Step-by-step reproduction instructions
* Proof-of-concept code (if available)
* Impact assessment (including robot, simulator, or network reachability)

**Detailed reports help NVIDIA evaluate and address issues faster.**

NVIDIA's Product Security Incident Response Team (PSIRT) will acknowledge
receipt, validate severity, develop fixes, and publish security bulletins as
appropriate.

## Security Architecture & Context

Isaac OS is a Holoscan-based Physical AI robotics platform (C++/Python/CUDA,
CMake). This repository contains reference **applications**, reusable
**operators/subgraphs**, FlatBuffers **messages**, and tools for sim-first
development and on-robot deployment (including Unitree G1/H2, SO-101,
GR00T/WBC stacks, an Isaac Sim UCX bridge, and metrics tooling).

This software operates at the **Application / SDK / CLI** level. Its primary
security responsibility is to keep robot command and state paths, simulation
transports, and operator-supplied configs under the operator's deployment
controls. It is not a multi-tenant internet service and does not implement
application-level authentication on UCX or Unitree DDS.

**Repository Exposure Classification:** Public.
Basis: this document is written for public consumption.

**Service Exposure Classification:** External / Regulated (high confidence).
Basis: sim-to-real robotics reference stack with real-robot command paths.

Key boundaries and interfaces:

* **Isaac Sim UCX:** `tools/isaac-sim-ucx` starts a UCX listener (default port
  50008) inside Isaac Sim. Isaac OS UCX endpoints in this tree are
  **client-only**. UCX messages are not authenticated or encrypted in these
  components.
* **Unitree DDS:** Unitree operators use SDK2 channel publishers/subscribers
  (DDS domain and network interface parameters). This repository does not add
  a DDS-security layer on top of that stack.
* **Robot command path:** fused operators and WBC/GR00T apps publish joint
  commands and related state. Limited input sanitization exists in places
  (for example SO-101 reach action clipping) but is not a complete safety or
  authentication boundary.
* **Operator CLI / files:** launchers, YAML/JSON app metadata, USD scenes, and
  metrics CLIs take local paths and flags. There is no in-app user identity
  model. The `./isaac_os` launcher runs the development container with
  `--privileged` and host PID/IPC namespaces so GPU, sensors, and shared
  memory work.
* **Foxglove visualization:** converters under `tools/foxglove` are schema
  plugins for visualization clients, not a hosted service. Separately, the
  MCAP logger (`common/mcap_logger`) can optionally host a Foxglove
  WebSocket server inside the application process to stream recorded topics
  live. It is disabled by default and is an unauthenticated, read-only
  observation path when enabled — see "Foxglove live streaming" below.

### Threat Model

The following scenarios represent the primary security concerns for this
project:

1. **Untrusted scene, config, or model files:** USD stages, YAML configs, and
   local model or engine files are loaded with the privileges of the Holoscan
   or Isaac Sim process. In particular, `./isaac_sim` pre-authorizes Python
   ScriptNodes embedded in USD scenes by default. A file from an untrusted
   source can therefore execute arbitrary Python with the privileges of the
   Isaac Sim process, or consume unbounded GPU and host memory.
2. **Untrusted recordings and visualization data:** Replayed logs and schema
   converters may allocate GPU or host buffers from dimensions in the file or
   message. Treat recordings the same way as other operator-supplied inputs.
3. **Shared-host interference:** Local IPC and on-disk inference caches assume
   a dedicated machine. Another local user can interfere if the same host is
   shared.

### Critical Security Assumptions

* UCX and Unitree DDS run on a **trusted, isolated network** (or equivalent
  controls). This code does not provide TLS, mTLS, or DDS Security.
  Payloads on those transports are treated as trusted sensor and command data.
* The MCAP logger's optional Foxglove live streaming, when an operator enables
  it, likewise assumes a **trusted, isolated network**: it authenticates no
  peer and encrypts no traffic, so every host that can reach its port is
  treated as an authorized observer of the recorded topics.
* Callers and adjacent stacks (Isaac Sim, Unitree SDK2, Holoscan, the host
  OS) are trusted; there is no multi-tenant isolation inside an application
  graph.
* Robot functional safety (estops, torque limits, collision checking) is
  **not** guaranteed by this repository; operators must enforce it in the
  robot/controller stack.
* Operator-supplied USD scenes, configs, recordings, and local model/engine
  caches are trusted on the machine that runs the application.
* The `./isaac_os` development container is run privileged on a dedicated
  trusted host.

## Deployment Security Guidance

Isaac OS reference applications and transport operators are not secure by
default on untrusted networks. Real deployments should add network isolation,
transport security, access control, and final command safety validation for
their environment.

### USD ScriptNodes

`./isaac_sim` (through `tools/isaac-sim-ucx/isaac_sim_ucx.py`) pre-authorizes
USD ScriptNode execution by default and disables Isaac Sim's consent dialog.
This supports unattended simulation workflows, but it also means that Python
embedded in a loaded USD scene executes automatically with the full privileges
of the Isaac Sim process.

Only load USD scenes from trusted sources. When opening a scene whose provenance
or integrity is uncertain, pass `--require-scriptnode-consent` to leave Isaac Sim's
ScriptNode consent gate enabled:

```console
./isaac_sim run --scene path/to/scene.usd -- --require-scriptnode-consent
```

In headless or otherwise unattended environments, the consent gate may prevent
ScriptNodes required by the scene from running. Review and trust the scene
before using the default auto-consent behavior.

### UCX

The UCX transport used with `operators/ucxx_ops` and `tools/isaac-sim-ucx` is
intended for trusted local or isolated simulation networks. Current Isaac Sim
workflows use Isaac Sim as the UCX server and Isaac OS as the UCX client.
Protect UCX simulation ports with loopback-only topology where available, host
firewall rules, VLAN isolation, VPN, or equivalent controls. `UCX_NET_DEVICES`
(and related UCX environment variables) can reduce accidental transport
exposure. Isaac OS UCX endpoints are client-only. Isaac Sim's UCX server does
not expose peer filtering or bind-address controls today.

### DDS

The Unitree DDS integrations are reference application paths for Unitree
robots and simulators. They do not add security controls on top of the
underlying Unitree SDK2 DDS communication. Deployments that require DDS
security must configure it in the DDS/robot network environment or add an
application-specific secure gateway.

### Foxglove live streaming

The MCAP logger (`common/mcap_logger`) can stream the topics it
records to a Foxglove client over a WebSocket server it hosts in the
application process. This is **off by default** (`enable_stream = false`);
nothing listens on any port unless an application explicitly opts in.

When enabled, the server binds `websocket_host`:`websocket_port`, which default
to **`127.0.0.1:8765`**, matching the Foxglove SDK's own default. Only clients
in the same network namespace as the application process can reach the port
without further action; a client sharing that namespace by virtue of host
networking reaches it directly. Reaching it from a separate namespace — for
example, a client on the host when the application runs in a container with
its own (bridged) network namespace, or a client on a different machine
entirely — requires either forwarding the port deliberately (for example over
SSH: `ssh -L 8765:localhost:8765 <user>@<host>`, once the port is itself
reachable from the SSH target) or an operator explicitly overriding
`websocket_host` to `0.0.0.0` or another externally reachable interface.
Overriding the bind address widens the exposure: the port becomes reachable
from anywhere that can route to the machine.

That listener provides **no authentication, no authorization, and no transport
encryption** (plaintext `ws://`), regardless of bind address. Any peer that can
reach the port may subscribe to every topic the application exports, which for
a robot application typically includes joint states, poses, transforms, camera
calibration, and image streams.

The exposure is **read-only**. The server advertises no
`WebSocketServerCapabilities` and registers no callbacks, so connected clients
cannot publish messages, set parameters, call services, or fetch assets. It is
an observation path, not a command path into the robot.

Deployments should therefore:

* Leave `enable_stream` off outside of development and debugging.
* Leave `websocket_host` at its `127.0.0.1` default and forward the port
  deliberately (for example over SSH) when a remote client needs it, rather
  than binding all interfaces.
* Restrict the port with host firewall rules, VLAN isolation, or a VPN when a
  non-loopback bind is genuinely required, and do not publish the container's
  port to untrusted networks.
* Treat the stream as sensitive telemetry: it carries the same data as the
  `.mcap` recording, live and unauthenticated.

### Privileged containers

`./isaac_os` uses `--privileged` and host PID/IPC namespaces. Use that launcher
only on a dedicated trusted machine. Do not expose the container to untrusted
users on a shared host.

## NVIDIA Product Security

For all security-related concerns, please visit NVIDIA's Product Security
portal at <https://www.nvidia.com/en-us/security>.
