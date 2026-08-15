def get_segment_paths(data_dir):
    segments_to_paths = {}

    for folder in data_dir.iterdir():
        for segment in folder.iterdir():
            if segment.is_dir():
                segments_to_paths[segment.name] = segment

    return segments_to_paths