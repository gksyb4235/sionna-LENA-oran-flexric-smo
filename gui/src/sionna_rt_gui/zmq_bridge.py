"""
zmq_bridge.py
─────────────────────────────────────────────────────────────────────
외부 Python (integration_loop.py) → GUI 실시간 브리지.

외부에서 ZMQ로 다음을 전송할 수 있다:
  - UE 위치 업데이트 → GUI에서 RX 포인트가 실시간 이동
  - OAI KPI (MCS/BLER/PRB) → GUI 패널에 오버레이
  - CIR 데이터 → GUI CIR 플롯 업데이트

포트:
  ZMQ_UE_PORT  = 5600  : UE 위치 수신 (REP)
  ZMQ_KPI_PORT = 5601  : OAI KPI 수신 (SUB)

메시지 포맷 (JSON):
  UE 위치:
    {"type": "ue_position", "name": "ue1", "position": [x, y, z]}
  시뮬레이션 상태(권장, tick당 1회):
    {"type": "simulation_state", "schema_version": 1,
     "ues": [{"name": "ue1", "position": [x, y, z]}],
     "gnbs": [{"name": "gnb1", "position": [x, y, z],
               "bearing_deg": 0, "tilt_deg": 0, "num_v": 1}]}
  KPI:
    {"type": "kpi", "ues": [{"rnti": 123, "dl_mcs": 13, "dl_bler": 0.0, ...}]}
  궤적 전체:
    {"type": "trajectory", "name": "ue1", "waypoints": [[x,y,z], ...], "velocity": 1.5}
  gNB 위치:
    {"type": "gnb_position", "name": "gnb", "position": [x, y, z]}
  gNB RET/안테나 상태 (ns3sionna_server.py의 node_antenna와 동일 규약):
    {"type": "gnb_antenna", "name": "gnb1", "bearing_deg": 90.0,
     "tilt_deg": 12.0, "num_v": 8}
"""

import json
import threading
import time
from typing import TYPE_CHECKING

import zmq

if TYPE_CHECKING:
    from .gui import SionnaRtGui

ZMQ_UE_PORT  = 5600
ZMQ_KPI_PORT = 5601


class ZMQBridge:
    """
    GUI 인스턴스에 붙어서 ZMQ 메시지를 처리하는 브리지.

    GUI tick() 루프에서 poll_and_apply()를 매 프레임 호출해야 한다.
    실제 Sionna/Polyscope 업데이트는 메인 스레드에서만 해야 하므로
    수신된 명령을 큐에 쌓고 메인 스레드에서 처리한다.
    """

    def __init__(self, ue_port: int = ZMQ_UE_PORT, kpi_port: int = ZMQ_KPI_PORT):
        self.ue_port  = ue_port
        self.kpi_port = kpi_port
        self._ctx     = None
        self._ue_sock = None
        self._kpi_sock = None

        self._cmd_queue: list  = []
        self._queue_lock = threading.Lock()

        # KPI 최신값 (ImGui 패널용)
        self.latest_kpi: list = []
        self.kpi_timestamp: float = 0.0
        # simulation_state가 관리하는 UE만 inactive 시 scene에서 제거한다.
        # GUI에서 사용자가 직접 만든 RX는 건드리지 않는다.
        self._managed_ue_names: set[str] = set()
        self._gnb_antenna_state: dict[str, tuple[float, float, int]] = {}

        self._running = False
        self._thread  = None

    def start(self):
        self._ctx      = zmq.Context()

        # UE 위치: REP 소켓 (요청-응답)
        self._ue_sock  = self._ctx.socket(zmq.REP)
        self._ue_sock.setsockopt(zmq.RCVTIMEO, 10)   # 10ms timeout
        self._ue_sock.bind(f"tcp://*:{self.ue_port}")

        # KPI: SUB 소켓 (pub-sub)
        self._kpi_sock = self._ctx.socket(zmq.SUB)
        self._kpi_sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._kpi_sock.setsockopt(zmq.RCVTIMEO, 10)
        self._kpi_sock.bind(f"tcp://*:{self.kpi_port}")

        self._running  = True
        self._thread   = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[ZMQBridge] 시작: UE포트={self.ue_port}, KPI포트={self.kpi_port}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._ue_sock:
            self._ue_sock.close()
        if self._kpi_sock:
            self._kpi_sock.close()
        if self._ctx:
            self._ctx.term()

    def _recv_loop(self):
        """백그라운드: ZMQ 수신 → 큐에 적재."""
        while self._running:
            # UE 위치 수신
            try:
                raw = self._ue_sock.recv_string()
                msg = json.loads(raw)
                with self._queue_lock:
                    self._cmd_queue.append(msg)
                self._ue_sock.send_string("OK")
            except zmq.Again:
                pass
            except Exception as e:
                try:
                    self._ue_sock.send_string(f"ERR:{e}")
                except Exception:
                    pass

            # KPI 수신
            try:
                raw = self._kpi_sock.recv_string()
                msg = json.loads(raw)
                if msg.get("type") == "kpi":
                    self.latest_kpi     = msg.get("ues", [])
                    self.kpi_timestamp  = time.time()
            except zmq.Again:
                pass
            except Exception:
                pass

    def poll_and_apply(self, gui: "SionnaRtGui"):
        """
        메인 스레드에서 호출: 큐에 쌓인 명령을 처리해서 GUI에 반영.
        gui.tick() 내부에서 호출된다.
        """
        with self._queue_lock:
            cmds = list(self._cmd_queue)
            self._cmd_queue.clear()

        for msg in cmds:
            try:
                self._apply(gui, msg)
            except Exception as e:
                print(f"[ZMQBridge] 명령 처리 오류: {e} — {msg}")

    def _apply(self, gui: "SionnaRtGui", msg: dict):
        """실제 GUI 상태 변경 (메인 스레드)."""
        t = msg.get("type")

        if t == "simulation_state":
            self._apply_simulation_state(gui, msg)

        elif t == "ue_position":
            # UE 위치 즉시 업데이트
            name = msg["name"]
            pos  = msg["position"]
            obj  = None
            try:
                obj = gui.scene.get(name)
            except Exception:
                pass

            if obj is None:
                # ns-3/server와 같은 이름을 사용해 다음 update가 같은 UE를 찾게 한다.
                gui.add_radio_device(pos, is_transmitter=False, name=name)
            else:
                obj.position = pos
                # Polyscope 포인트 업데이트
                from .sionna_utils import set_or_update_radio_devices_polyscope
                set_or_update_radio_devices_polyscope(
                    gui.scene.receivers, is_transmitter=False, gui=gui
                )
                # 경로 자동 업데이트
                if gui.cfg.paths.auto_update:
                    gui.update_paths(show=True)

        elif t == "gnb_position":
            # gNB(TX) 위치 업데이트
            name = msg["name"]
            pos  = msg["position"]
            try:
                obj = gui.scene.get(name)
                obj.position = pos
                from .sionna_utils import set_or_update_radio_devices_polyscope
                set_or_update_radio_devices_polyscope(
                    gui.scene.transmitters, is_transmitter=True, gui=gui
                )
                if gui.cfg.paths.auto_update:
                    gui.update_paths(show=True)
            except Exception:
                gui.add_radio_device(pos, is_transmitter=True, name=name)

        elif t == "gnb_antenna":
            gui.set_gnb_antenna_state(
                msg["name"], msg["bearing_deg"], msg["tilt_deg"], int(msg["num_v"])
            )

        elif t == "trajectory":
            # 궤적 전체를 animation_config에 등록
            name      = msg["name"]
            waypoints = msg["waypoints"]
            velocity  = msg.get("velocity", 1.11)

            from .animation import Trajectory
            traj = gui.animation_config.trajectories[name]
            traj.clear()
            for wp in waypoints:
                traj.add_point(wp)
            traj.velocity = velocity
            traj.enabled  = True
            # 재생 시작
            gui.animation_config.playing = True
            print(f"[ZMQBridge] 궤적 등록: {name}, {len(waypoints)} 포인트")

        elif t == "animation_control":
            # 재생/일시정지/속도
            action = msg.get("action")
            if action == "play":
                gui.animation_config.playing = True
            elif action == "pause":
                gui.animation_config.playing = False
            elif action == "speed":
                gui.animation_config.speed_multiplier = float(msg.get("speed", 1.0))

        elif t == "radio_map":
            # Radio Map 계산 요청
            if msg.get("compute", False):
                rm = gui.compute_radio_map()
                if rm:
                    gui.set_radio_map(rm, show=True)

    def _apply_simulation_state(self, gui: "SionnaRtGui", msg: dict):
        """Apply one complete active-UE/gNB state with one Polyscope refresh."""
        ues = msg.get("ues", [])
        gnbs = msg.get("gnbs", [])
        if not isinstance(ues, list) or not isinstance(gnbs, list):
            raise ValueError("simulation_state ues/gnbs must be arrays")

        active_names: set[str] = set()
        for item in ues:
            name = str(item["name"])
            position = item["position"]
            active_names.add(name)
            obj = gui.scene.get(name)
            if obj is None:
                gui.add_radio_device(
                    position,
                    is_transmitter=False,
                    allow_auto_update=False,
                    name=name,
                )
            else:
                obj.position = position

        for name in sorted(self._managed_ue_names - active_names):
            if gui.scene.get(name) is not None:
                gui.scene.remove(name)
        self._managed_ue_names = active_names

        for item in gnbs:
            name = str(item["name"])
            position = item["position"]
            obj = gui.scene.get(name)
            if obj is None:
                gui.add_radio_device(
                    position,
                    is_transmitter=True,
                    allow_auto_update=False,
                    name=name,
                )
            else:
                obj.position = position

            if {
                "bearing_deg",
                "tilt_deg",
                "num_v",
            }.issubset(item):
                state = (
                    float(item["bearing_deg"]),
                    float(item["tilt_deg"]),
                    int(item["num_v"]),
                )
                if self._gnb_antenna_state.get(name) != state:
                    gui.set_gnb_antenna_state(name, *state)
                    self._gnb_antenna_state[name] = state

        from .sionna_utils import set_or_update_radio_devices_polyscope
        set_or_update_radio_devices_polyscope(
            gui.scene.receivers,
            is_transmitter=False,
            gui=gui,
        )
        set_or_update_radio_devices_polyscope(
            gui.scene.transmitters,
            is_transmitter=True,
            gui=gui,
        )
        if gui.cfg.paths.auto_update:
            gui.update_paths(show=True)


# ─── 외부 클라이언트 (integration_loop.py에서 사용) ─────────────────
class ZMQBridgeClient:
    """
    integration_loop.py 또는 Jupyter에서 GUI에 명령을 보내는 클라이언트.

    사용 예:
        client = ZMQBridgeClient()
        client.connect()
        client.send_ue_position("ue1", [30.0, 20.0, 1.5])
        client.send_trajectory("ue1", [[0,0,1.5],[50,0,1.5],[50,50,1.5]])
        client.disconnect()
    """

    def __init__(self, host: str = "localhost",
                 ue_port: int = ZMQ_UE_PORT,
                 kpi_port: int = ZMQ_KPI_PORT):
        self.host     = host
        self.ue_port  = ue_port
        self.kpi_port = kpi_port
        self._ctx     = None
        self._ue_sock = None
        self._kpi_sock = None

    def connect(self):
        self._ctx      = zmq.Context()
        self._ue_sock  = self._ctx.socket(zmq.REQ)
        self._ue_sock.setsockopt(zmq.SNDTIMEO, 1000)
        self._ue_sock.setsockopt(zmq.RCVTIMEO, 1000)
        self._ue_sock.connect(f"tcp://{self.host}:{self.ue_port}")

        self._kpi_sock = self._ctx.socket(zmq.PUB)
        self._kpi_sock.connect(f"tcp://{self.host}:{self.kpi_port}")
        time.sleep(0.1)  # PUB-SUB 연결 안정화
        print(f"[ZMQBridgeClient] 연결: {self.host}:{self.ue_port}/{self.kpi_port}")

    def disconnect(self):
        if self._ue_sock:
            self._ue_sock.close()
        if self._kpi_sock:
            self._kpi_sock.close()
        if self._ctx:
            self._ctx.term()

    def _send_cmd(self, msg: dict) -> bool:
        try:
            self._ue_sock.send_string(json.dumps(msg))
            resp = self._ue_sock.recv_string()
            return resp == "OK"
        except zmq.Again:
            return False

    def send_ue_position(self, name: str, position: list) -> bool:
        return self._send_cmd({
            "type": "ue_position",
            "name": name,
            "position": position,
        })

    def send_simulation_state(
        self,
        ues: list,
        gnbs: list | None = None,
        sim_time_s: float | None = None,
    ) -> bool:
        """Send one complete active-UE set.

        Receivers previously sent by this batched API but omitted from
        ``ues`` are removed from the GUI. Legacy single-position calls keep
        their original independent behavior.
        """

        message = {
            "type": "simulation_state",
            "schema_version": 1,
            "ues": ues,
            "gnbs": gnbs or [],
        }
        if sim_time_s is not None:
            message["sim_time_s"] = float(sim_time_s)
        return self._send_cmd(message)

    def send_gnb_position(self, name: str, position: list) -> bool:
        return self._send_cmd({
            "type": "gnb_position",
            "name": name,
            "position": position,
        })

    def send_gnb_antenna(self, name: str, bearing_deg: float, tilt_deg: float,
                        num_v: int) -> bool:
        return self._send_cmd({
            "type": "gnb_antenna",
            "name": name,
            "bearing_deg": bearing_deg,
            "tilt_deg": tilt_deg,
            "num_v": num_v,
        })

    def send_trajectory(self, name: str, waypoints: list,
                        velocity: float = 1.11) -> bool:
        return self._send_cmd({
            "type":      "trajectory",
            "name":      name,
            "waypoints": waypoints,
            "velocity":  velocity,
        })

    def send_kpi(self, ue_stats: list):
        """
        ue_stats: [{"rnti": 123, "dl_mcs": 13, "dl_bler": 0.0, ...}, ...]
        """
        msg = json.dumps({"type": "kpi", "ues": ue_stats})
        try:
            self._kpi_sock.send_string(msg)
        except Exception:
            pass

    def play(self, speed: float = None):
        msg = {"type": "animation_control", "action": "play"}
        if speed is not None:
            msg["action"] = "speed"
            msg["speed"]  = speed
        return self._send_cmd(msg)

    def pause(self):
        return self._send_cmd({"type": "animation_control", "action": "pause"})

    def request_radio_map(self):
        return self._send_cmd({"type": "radio_map", "compute": True})
