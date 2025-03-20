# Extract coordinates with size filtering for StarDist channels
def extract_coordinates_with_size_filter(labels, min_area, max_area):
    from skimage.measure import regionprops
    properties = regionprops(labels)
    coordinates = []
    for prop in properties:
        if min_area <= prop.area <= max_area:
            coordinates.append((int(prop.centroid[1]), int(prop.centroid[0])))
    return coordinates
