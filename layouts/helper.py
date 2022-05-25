from os import walk


def get_room_files(rooms_dir):
    (_, _, room_files) = next(walk(f"layouts/{rooms_dir}"), (None, None, []))
    room_files.sort()
    if ".DS_Store" in room_files:
        room_files.remove(".DS_Store")
    return room_files


def get_room_path(rooms_dir, room_file):
    return f"layouts/{rooms_dir}/{room_file}"
