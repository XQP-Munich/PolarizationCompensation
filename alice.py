from QKD.Worker import Client
import time
from PyQt5.QtCore import (
    QReadWriteLock,
    QThread,
    QObject,
    pyqtSignal,
    QRunnable,
    pyqtSlot,
    QThreadPool,
)

threadpool = QThreadPool.globalInstance()
client = Client()
threadpool.start(client)
while client.is_running:
    print(".")
    time.sleep(1)
