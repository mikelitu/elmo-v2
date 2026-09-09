import requests
import socket

MAX_ERROR_COUNT = 5


def connect(ip, port=8001, timeout=2):
    """
    Connect to a robot with a static IP from WSL.
    """
    address = f"http://{ip}:{port}"

    try:
        # Simple TCP reachability check
        with socket.create_connection((ip, port), timeout=timeout):
            pass

        return True, "OK", Robot(address)

    except Exception as e:
        return False, str(e), None

def set_robot_model(model):
        robot_model = model

class Robot:
    def __init__(self, address):
        self.address = address
        self.ip = address.split("//")[1].split(":")[0]
        self.error_count = 0

    def update_status(self):
        try:
            r = requests.get(self.address + "/status", timeout=1)
            r.raise_for_status()

            status = r.json()
            for k, v in status.items():
                setattr(self, k, v)

            self.error_count = 0

        except Exception as e:
            self.error_count += 1
            print("Status error:", e)

            if self.error_count > MAX_ERROR_COUNT:
                self.on_disconnect()

    def send_command(self, command, **kwargs):
        try:
            payload = {"op": command, **kwargs}

            r = requests.post(
                self.address + "/command",
                json=payload,
                timeout=1
            )
            r.raise_for_status()

            res = r.json()
            if not res.get("success", True):
                self.on_error(res.get("message", "Unknown error"))

        except Exception as e:
            print("Command error:", e)

    def on_error(self, message):
        print("Robot error:", message)

    def on_disconnect(self):
        print(f"Connection to robot lost ({self.ip})")
    

if __name__ == "__main__":
    # Example usage
    success, message, robot = connect("192.168.0.4", 8001)
    print(success, message)
    if success:
        robot.update_status()
        print(robot.__dict__)
        robot.send_command("play_sound", name="saludo.mp3")