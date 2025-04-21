import os
import pandas as pd
import datetime

def test_csv_write():
    """
    Test function to verify if we can create and save custom CSV files on the server.
    Creates a simple CSV file with test data and verifies its existence.
    """
    # Create test data
    test_data = {
        'id': [1, 2, 3, 4, 5],
        'value': [10, 20, 30, 40, 50],
        'timestamp': [datetime.datetime.now() for _ in range(5)]
    }
    
    # Create DataFrame
    df = pd.DataFrame(test_data)
    
    # Define output path
    output_file = 'test_output.csv'
    
    # Save CSV file
    df.to_csv(output_file, index=False)
    
    # Verify file was created
    if os.path.exists(output_file):
        print("SUCCESS: CSV file '{}' was created successfully".format(output_file))
        print("File size: {} bytes".format(os.path.getsize(output_file)))
        
        # Read back and verify content
        read_df = pd.read_csv(output_file)
        print("File contains {} rows and {} columns".format(len(read_df), len(read_df.columns)))
        print("File preview:")
        print(read_df.head())
    else:
        print("ERROR: Failed to create CSV file '{}'".format(output_file))
    
    return os.path.exists(output_file)

if __name__ == "__main__":
    print("Testing CSV write functionality...")
    result = test_csv_write()
    print("Test {}".format("passed" if result else "failed")) 