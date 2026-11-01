FROM tiryoh/ros2-desktop-vnc@sha256:b9eed044a3a3ab090081cd2126a10cfe952175ebfabf5bc48eb3944bd1679158

ARG SJTU_REV=3f8627ebddf390fd35ffb84b8654b62e0b66fca2
RUN --mount=type=secret,id=corporate_ca \
    if [ -f /run/secrets/corporate_ca ]; then \
      openssl x509 -inform DER -in /run/secrets/corporate_ca -out /usr/local/share/ca-certificates/local-build-ca.crt \
      && update-ca-certificates; \
    fi
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3-venv ros-humble-xacro ros-humble-cv-bridge \
    ros-humble-tf2-geometry-msgs ros-humble-joy ros-humble-joint-state-publisher \
    ros-humble-robot-state-publisher ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
COPY pid-initialization.patch /opt/sos/pid-initialization.patch
RUN git clone --branch ros2 --single-branch https://github.com/NovoG93/sjtu_drone.git /opt/drone/src/sjtu_drone \
    && cd /opt/drone/src/sjtu_drone && git checkout ${SJTU_REV} \
    && git apply /opt/sos/pid-initialization.patch \
    && /bin/bash -c 'source /opt/ros/humble/setup.bash && cd /opt/drone && colcon build --packages-select sjtu_drone_description sjtu_drone_bringup --cmake-args -DBUILD_TESTING=OFF'
COPY requirements-train.txt /opt/sos/requirements-train.txt
RUN python3 -m venv --system-site-packages /opt/sos/train \
    && /opt/sos/train/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && /opt/sos/train/bin/pip install -r /opt/sos/requirements-train.txt
WORKDIR /workspace/sos
COPY scripts/prepare_workspace.py /opt/sos/prepare_workspace.py
RUN python3 -c 'from pathlib import Path; p = Path("/entrypoint.sh"); text = p.read_text(); anchor = "exec /bin/tini -- supervisord -n -c /etc/supervisor/supervisord.conf"; assert text.count(anchor) == 1, "Unexpected desktop entrypoint"; p.write_text(text.replace(anchor, "/usr/bin/python3 /opt/sos/prepare_workspace.py || exit $?\n" + anchor))'
