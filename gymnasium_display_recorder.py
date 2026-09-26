
# pyrefly: ignore [missing-import]
import gymnasium as gym

# pyrefly: ignore [missing-import]
from gymnasium.wrappers import RecordVideo
import glob
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay
try:
    # pyrefly: ignore [missing-import]
    from pyvirtualdisplay import Display
    import os
    os.environ['PYVIRTUALDISPLAY_DISPLAYFD'] = '0'
    # Khởi động màn hình ảo
    display = Display(visible=0, size=(640, 480))
    display.start()
except Exception:
    display = None

def gym_make(env_name, run_name, render_fps=30):
    """
    Tạo môi trường Gym và bật ghi hình video.

    :param env_name: Tên môi trường.
    :param run_name: Tên của lần chạy (dùng làm tiền tố tên file video).
    :param render_fps: Số khung hình hiển thị mỗi giây.
    """
    env = gym.make(env_name, render_mode="rgb_array")
    env.metadata['render_fps'] = render_fps

    env = RecordVideo(env, 
                      video_folder='./videos', 
                      episode_trigger=lambda episode_id: True,
                      name_prefix=f'video_{run_name}')
    
    #env.reset()
    return env

def show(run_name, episode_id=0):
    """
    Hiển thị video đã ghi trong notebook Jupyter.

    :param run_name: Tên của lần chạy (dùng để tìm file video tương ứng).
    :param episode_id: Số thứ tự tập (episode) cần hiển thị, mặc định là 0.
    """
    video = io.open(glob.glob(f'./videos/video_{run_name}*{episode_id}.mp4')[0], 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''
        <video width="640" height="480" controls>
            <source src="data:video/mp4;base64,{0}" type="video/mp4" />
        </video>
    '''.format(encoded.decode('ascii'))))