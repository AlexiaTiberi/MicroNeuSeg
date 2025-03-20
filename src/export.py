
def save_coordinates_to_csv(coordinates, output_path):
    import pandas as pd
    df = pd.DataFrame(coordinates, columns=["x", "y"])
    df.to_csv(output_path, index=False)
    print(f"Coordinates saved to: {output_path}")