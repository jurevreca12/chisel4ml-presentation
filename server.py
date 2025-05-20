from chisel4ml.chisel4ml_server import Chisel4mlServer
import subprocess
import tempfile
import socket
from pathlib import Path


def get_free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def create_server(c4ml_jar):
    tmp_dir = Path(tempfile.TemporaryDirectory(prefix="chisel4ml").name)
    c4ml_jar = Path(c4ml_jar).resolve()
    free_port = get_free_port()
    assert c4ml_jar.exists()
    command = ["java", "-jar", f"{c4ml_jar}", "-p", f"{free_port}", "-d", f"{tmp_dir}"]
    c4ml_subproc = subprocess.Popen(command)
    c4ml_server = Chisel4mlServer(tmp_dir, free_port)
    return c4ml_server, c4ml_subproc