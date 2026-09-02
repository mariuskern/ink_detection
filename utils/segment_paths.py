def get_segment_paths(data_dir):
    segments_to_paths = {}

    for scroll in data_dir.iterdir():
        for category in scroll.iterdir():
            for segment in category.iterdir():
                if segment.is_dir():
                    segments_to_paths[segment.name] = segment

    return segments_to_paths