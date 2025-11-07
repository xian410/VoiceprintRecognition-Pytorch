import threading
import ctypes
import soundcard
from yeaudio.audio import AudioSegment

# === COM 初始化常量和函数 ===
COINIT_APARTMENTTHREADED = 0x2

def _com_initialize():
    """在当前线程初始化 COM"""
    ole32 = ctypes.windll.ole32
    hr = ole32.CoInitializeEx(0, COINIT_APARTMENTTHREADED)
    # S_OK = 0, RPC_E_CHANGED_MODE = 0x80070057（已初始化过，可忽略）
    if hr != 0 and hr != 0x80070057:
        raise RuntimeError(f"CoInitializeEx failed with HRESULT: 0x{hr & 0xFFFFFFFF:X}")

def _com_uninitialize():
    """反初始化 COM"""
    try:
        ctypes.windll.ole32.CoUninitialize()
    except:
        pass  # 忽略异常
# ============================


class AudioPlayer:
    def __init__(self, audio_path):
        """音频播放器

        Args:
            audio_path (str): 音频文件路径
        """
        self.playing = False
        self.to_pause = False
        self.pos = 0
        self.audio_segment = AudioSegment.from_file(audio_path)
        self.samples = self.audio_segment.samples
        self.sample_rate = self.audio_segment.sample_rate
        self.default_speaker = soundcard.default_speaker()
        self.block_size = self.sample_rate // 2

    def _play(self):
        # === 关键修复：在线程开始时初始化 COM ===
        _com_initialize()
        try:
            self.to_pause = False
            self.playing = True
            with self.default_speaker.player(samplerate=self.sample_rate) as p:
                start_index = int(self.pos * self.sample_rate)
                for i in range(start_index, len(self.samples), self.block_size):
                    if self.to_pause:
                        break
                    self.pos = i / self.sample_rate
                    p.play(self.samples[i:i + self.block_size])
        finally:
            _com_uninitialize()
            self.playing = False

    def play(self):
        if not self.playing:
            thread = threading.Thread(target=self._play)
            thread.start()

    def pause(self):
        self.to_pause = True

    def seek(self, seconds=0.0):
        self.pos = seconds

    def current_time(self):
        return self.pos