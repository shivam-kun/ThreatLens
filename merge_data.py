# merge_data.py
import pandas as pd
import sqlite3

def generate_aews_feature():
    try:
        # Connect to the database created by aews_monitor.py
        conn = sqlite3.connect('threat_feed.db')
        
        # Aggregate high-level alerts by day
        # The strftime function extracts the 'YYYY-MM-DD' part of the timestamp
        query = "SELECT strftime('%Y-%m-%d', timestamp) as date, COUNT(*) as high_alert_volume FROM alerts WHERE threat_level IN ('High', 'Medium') GROUP BY date"
        aews_df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert the 'date' column to datetime objects
        aews_df['date'] = pd.to_datetime(aews_df['date'])
        
        print("AEWS daily alert volumes:")
        print(aews_df.head())
        
        # Load your main forecast dataset
        forecast_df = pd.read_csv('datasets/forecast.csv')
        
        # Create a 'date' column from the 'Year' column for merging
        forecast_df['date'] = pd.to_datetime(forecast_df['Year'].astype(str) + '-01-01')

        # Merge the two dataframes
        # A left merge ensures you keep all your original forecast data
        merged_df = pd.merge(forecast_df, aews_df, on='date', how='left')
        
        # Fill any days with no alerts with 0
        merged_df['high_alert_volume'].fillna(0, inplace=True)
        
        # Drop the temporary date column if you don't need it
        merged_df.drop('date', axis=1, inplace=True)

        # Save the new, enriched dataset
        output_path = 'datasets/unified_training_data.csv'
        merged_df.to_csv(output_path, index=False)
        
        print(f"\nSuccessfully created enriched dataset at: {output_path}")
        print(merged_df.head())
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    generate_aews_feature()