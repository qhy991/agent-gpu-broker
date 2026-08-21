# Root-managed team deployment

Run the broker as a dedicated unprivileged account, never as root. Every submitted
command executes as the daemon account, so that account must contain no personal
credentials and must not belong to `sudo`, `docker`, or other privileged groups.

## 1. Create the service identity and trusted client group

```bash
groupadd --system gpuq-users
useradd --system --gid gpuq-users \
  --home-dir /var/lib/agent-gpu-broker \
  --shell /usr/sbin/nologin gpuq

usermod -aG gpuq-users ALICE
usermod -aG gpuq-users BOB
```

Users must start a new login session after group membership changes. Membership
in `gpuq-users` grants the ability to execute arbitrary commands as `gpuq`; add
only mutually trusted team members.

## 2. Install the root-owned program

```bash
git clone REPOSITORY_URL /opt/agent-gpu-broker
python3 -m venv /opt/agent-gpu-broker/.venv
/opt/agent-gpu-broker/.venv/bin/pip install -e /opt/agent-gpu-broker
chown -R root:root /opt/agent-gpu-broker
chmod -R a+rX /opt/agent-gpu-broker

ln -s /opt/agent-gpu-broker/.venv/bin/gpuq /usr/local/bin/gpuq
ln -s /opt/agent-gpu-broker/.venv/bin/gpu-run /usr/local/bin/gpu-run
```

The service account also needs normal read/write access to the team's shared
working directories. Do not grant it access to users' private home directories.
On systems where NVIDIA device files are restricted to `video` or `render`, add
`gpuq` only to the device-access group required by that system.

## 3. Install and start the system service

Ensure no user-level broker is still serving `/tmp/agent-gpu-broker.sock`; one
machine must have exactly one queue owner.

```bash
install -Dm644 \
  /opt/agent-gpu-broker/deploy/gpu-agent-broker.system.service \
  /etc/systemd/system/gpu-agent-broker.service

systemctl daemon-reload
systemctl enable --now gpu-agent-broker.service
systemctl status gpu-agent-broker.service
gpuq status
```

The service creates shared runtime and state directories and exposes the Unix
socket as `gpuq:gpuq-users` mode `0660`. It automatically starts at boot; login
lingering is not required. To coordinate with another local GPU scheduler,
configure both systems to use `/run/agent-gpu-broker/locks` and the same file
naming convention.

## Security boundary

This deployment is suitable only for a trusted team using shared workspaces. It
does not preserve each caller's Unix identity. If commands must run as individual
users, use a privileged authenticated per-user executor, containers, or a cluster
scheduler such as Slurm instead of exposing this command-execution daemon.
