# root 管理的团队部署

broker 必须使用专门的无特权账号运行，不能以 root 身份执行。所有提交的命令都会
继承 daemon 账号的身份，因此该账号不能保存个人凭据，也不能加入 `sudo`、
`docker` 等特权用户组。

## 1. 创建服务账号和可信客户端组

```bash
groupadd --system gpuq-users
useradd --system --gid gpuq-users \
  --home-dir /var/lib/agent-gpu-broker \
  --shell /usr/sbin/nologin gpuq

usermod -aG gpuq-users ALICE
usermod -aG gpuq-users BOB
```

用户加入组后需要重新登录。`gpuq-users` 成员能够以 `gpuq` 身份提交任意命令，
所以只能加入相互信任的团队成员。

## 2. 安装由 root 管理的程序

```bash
git clone REPOSITORY_URL /opt/agent-gpu-broker
python3 -m venv /opt/agent-gpu-broker/.venv
/opt/agent-gpu-broker/.venv/bin/pip install -e /opt/agent-gpu-broker
chown -R root:root /opt/agent-gpu-broker
chmod -R a+rX /opt/agent-gpu-broker

ln -s /opt/agent-gpu-broker/.venv/bin/gpuq /usr/local/bin/gpuq
ln -s /opt/agent-gpu-broker/.venv/bin/gpu-run /usr/local/bin/gpu-run
```

`gpuq` 服务账号还必须能够读写团队共享的工作目录，但不要授予其访问成员私人目录
的权限。如果 NVIDIA 设备文件只允许 `video` 或 `render` 组访问，仅将 `gpuq`
加入当前系统实际需要的设备访问组。

## 3. 安装并启动系统服务

启动前应确认没有用户级 broker 仍在监听 `/tmp/agent-gpu-broker.sock`；一台机器只能
存在一个队列所有者。

```bash
install -Dm644 \
  /opt/agent-gpu-broker/deploy/gpu-agent-broker.system.service \
  /etc/systemd/system/gpu-agent-broker.service

systemctl daemon-reload
systemctl enable --now gpu-agent-broker.service
systemctl status gpu-agent-broker.service
gpuq status
```

service 会创建共享运行目录和状态目录，Unix socket 权限为
`gpuq:gpuq-users 0660`。系统服务随开机启动，不需要 login lingering。若需要与
其他本机 GPU 调度器协作，应让它们共同使用 `/run/agent-gpu-broker/locks`，并采用
相同的锁文件命名规则。

## 安全边界

这种部署只适用于共享工作目录的可信团队，无法保留每个调用者的 Unix 身份。如果
命令必须以各成员自己的身份运行，应使用经过认证的按用户执行器、容器或 Slurm 等
集群调度器，而不是直接开放这个命令执行 daemon。
