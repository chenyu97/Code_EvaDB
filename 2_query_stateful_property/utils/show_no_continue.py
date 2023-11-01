import json
from pathlib import Path


resource_dir = Path(Path(__file__).parent, "resources")
file_path = (resource_dir / "test-tracks.json").as_posix()

with open(file_path, 'r') as file:
    data = json.load(file)
    for key, value in data.items():
        f_last = int(value["frames"][0][-10:-4])
        for index, frame_path in enumerate(value["frames"][1:]):
            f_cur = int(frame_path[-10:-4])
            if f_cur - f_last != 1:
                print(str(key) + ': ' + str(index))
                print(f_last)
                print(f_cur)
                break
            f_last = f_cur

