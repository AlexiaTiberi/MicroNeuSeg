# config.py
# Define channel-specific parameters
channel_params = {
    "Iba1": {
        "min_area": 30,
        "max_area": 500,
        "low_sigma":0.6,
        "high_sigma": 5,
        "solidity_thresh": 0.2,
        "eccentricity_thresh": 0.99,
        "min_distance": 20,
        "chambolle": False, #sometimes big processes close to the some get recognized as cells, use min_distance to filter out this - it refers as the minimum distance between 2 bb and will eliminate the smallest one
    },
    "PV": {
        "min_area": 80,
        "max_area": 600,
        "low_sigma": 0.7,
        "high_sigma": 5,
        "solidity_thresh": 0, 
        "eccentricity_thresh": 0.97 
    },
    "NeuN": {
        "prob_thresh": 0.1,
        "nms_thresh": 0.1,
        "min_area": 90,
        "max_area": 500
    }
}