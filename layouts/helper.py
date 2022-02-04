from os import walk


def get_room_files(rooms_dir):
    (_, _, room_files) = next(walk(f"layouts/{rooms_dir}"), (None, None, []))
    return room_files


def get_room_path(rooms_dir, room_file):
    return f"layouts/{rooms_dir}/{room_file}"
