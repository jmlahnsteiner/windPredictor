"""
Sailing Wind Prediction System

SETUP INSTRUCTIONS:
1. Create virtual environment:
   python3 -m venv sailing_env
   source sailing_env/bin/activate  # On Windows: sailing_env\\Scripts\\activate

2. Install required packages:
   pip install pandas numpy matplotlib seaborn scikit-learn openpyxl

3. Update file_path in main() function to point to your Excel file

REQUIRED PACKAGES:
- pandas>=1.3.0
- numpy>=1.20.0
- matplotlib>=3.3.0
- seaborn>=0.11.0
- scikit-learn>=1.0.0
- openpyxl>=3.0.0

USAGE:
python main.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class SailingWindPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.data = None

    def load_excel_data(self, file_path):
        """Load weather data from Excel file with multi-level headers"""
        try:
            # Read Excel file with multi-level headers
            df = pd.read_excel(file_path, header=[0, 1])

            print(f"Original multi-level columns in Excel file:")
            for i, col in enumerate(df.columns):
                print(f"  {i}: {col}")

            # Flatten multi-level columns by combining level 0 and level 1
            # Handle cases where level 0 might be 'Unnamed' or empty
            new_columns = []
            for col in df.columns:
                level0, level1 = col
                if pd.isna(level0) or 'Unnamed' in str(level0) or level0 == '':
                    new_columns.append(level1)
                elif pd.isna(level1) or level1 == '':
                    new_columns.append(level0)
                else:
                    new_columns.append(f"{level0}_{level1}")

            df.columns = new_columns

            print(f"\nFlattened columns:")
            for i, col in enumerate(df.columns):
                print(f"  {i}: '{col}'")

            # Enhanced column mapping for multi-level headers
            column_mapping = {
                'Außen_Temperature(℃)': 'temperature',
                'Außen_Feels Like(℃)': 'feels_like',
                'Außen_Dew Point(℃)': 'dew_point',
                'Außen_Humidity(%)': 'humidity',
                'Solar und UVI_Solar(W/m²)': 'solar',
                'Solar und UVI_UVI': 'uvi',
                'Regenfall_Rain Rate(mm/hr)': 'rain_rate',
                'Regenfall_Daily(mm)': 'daily_rain',
                'Regenfall_Event(mm)': 'event_rain',
                'Regenfall_Hourly(mm)': 'hourly_rain',
                'Regenfall_Weekly(mm)': 'weekly_rain',
                'Regenfall_Monthly(mm)': 'monthly_rain',
                'Regenfall_Yearly(mm)': 'yearly_rain',
                'Luftdruck_Relative(hPa)': 'pressure_relative',
                'Luftdruck_Absolute(hPa)': 'pressure_absolute',
                'Wassertemperatur_Temperature(℃)': 'water_temperature',
                'Wind_Wind Speed(knots)': 'wind_speed',
                'Wind_Wind Gust(knots)': 'wind_gust',
                'Wind_Wind Direction(º)': 'wind_direction',
                'Time': 'timestamp'
            }

            # Rename columns
            df = df.rename(columns=column_mapping)

            print(f"\nColumns after mapping:")
            for i, col in enumerate(df.columns):
                print(f"  {i}: '{col}'")

            # Auto-detect wind columns if mapping failed
            df = self._auto_detect_wind_columns(df)

            # Check for required wind columns
            required_cols = ['wind_speed', 'wind_direction']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"\nMissing required columns: {missing_cols}")
                print("Available columns:", list(df.columns))
                print("\nTrying fuzzy column matching...")
                df = self._fuzzy_column_matching(df)

                # Check again after fuzzy matching
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"Still missing columns after fuzzy matching: {missing_cols}")
                    return None

            # Parse timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                # Try to find timestamp column
                timestamp_col = df.columns[0]
                df['timestamp'] = pd.to_datetime(df[timestamp_col])

            # Set timestamp as index
            df.set_index('timestamp', inplace=True)

            # Remove any duplicate timestamps
            df = df[~df.index.duplicated(keep='first')]

            # Sort by timestamp
            df = df.sort_index()

            print(f"\nLoaded {len(df)} records from {file_path}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")

            return df

        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None

    def _fuzzy_column_matching(self, df):
        """Fuzzy matching for column names when exact mapping fails"""
        # Look for wind speed with partial matching
        if 'wind_speed' not in df.columns:
            for col in df.columns:
                col_lower = col.lower()
                if ('wind' in col_lower and 'speed' in col_lower) or \
                   ('wind' in col_lower and 'knot' in col_lower) or \
                   ('geschwindigkeit' in col_lower):
                    df['wind_speed'] = df[col]
                    print(f"Fuzzy matched wind speed column: '{col}'")
                    break

        # Look for wind direction with partial matching
        if 'wind_direction' not in df.columns:
            for col in df.columns:
                col_lower = col.lower()
                if ('wind' in col_lower and 'direction' in col_lower) or \
                   ('wind' in col_lower and 'richtung' in col_lower) or \
                   ('wind' in col_lower and '°' in col) or \
                   ('wind' in col_lower and 'grad' in col_lower):
                    df['wind_direction'] = df[col]
                    print(f"Fuzzy matched wind direction column: '{col}'")
                    break

        return df


    def _auto_detect_wind_columns(self, df):
        """Auto-detect wind columns if standard mapping fails"""
        # Common patterns for wind speed columns
        wind_speed_patterns = [
            'wind speed', 'windspeed', 'wind_speed', 'geschwindigkeit',
            'wind', 'windgeschwindigkeit', 'speed', 'knots'
        ]

        # Common patterns for wind direction columns
        wind_direction_patterns = [
            'wind direction', 'winddirection', 'wind_direction', 'richtung',
            'windrichtung', 'direction', 'dir', 'degrees', 'grad'
        ]

        # Check if wind columns already exist
        if 'wind_speed' in df.columns and 'wind_direction' in df.columns:
            return df

        # Find wind speed column
        if 'wind_speed' not in df.columns:
            for col in df.columns:
                if any(pattern in col.lower() for pattern in wind_speed_patterns):
                    if 'knot' in col.lower() or 'speed' in col.lower():
                        df['wind_speed'] = df[col]
                        print(f"Auto-detected wind speed column: '{col}'")
                        break

        # Find wind direction column
        if 'wind_direction' not in df.columns:
            for col in df.columns:
                if any(pattern in col.lower() for pattern in wind_direction_patterns):
                    if 'direction' in col.lower() or 'richtung' in col.lower() or '°' in col or 'grad' in col.lower():
                        df['wind_direction'] = df[col]
                        print(f"Auto-detected wind direction column: '{col}'")
                        break

        return df

    def create_wind_consistency_features(self, df):
        """Create features for wind direction consistency over 2-hour windows"""
        df = df.copy()

        # Calculate rolling statistics for wind direction consistency
        # Use 2-hour window (24 measurements at 5-minute intervals)
        window_size = 24

        # Wind direction consistency metrics
        df['wind_dir_std_2h'] = df['wind_direction'].rolling(
            window=window_size, min_periods=12).std()

        df['wind_dir_range_2h'] = df['wind_direction'].rolling(
            window=window_size, min_periods=12).apply(
            lambda x: self._circular_range(x) if len(x) > 0 else np.nan)

        # Wind speed features
        df['wind_speed_mean_2h'] = df['wind_speed'].rolling(
            window=window_size, min_periods=12).mean()

        df['wind_speed_std_2h'] = df['wind_speed'].rolling(
            window=window_size, min_periods=12).std()

        df['wind_speed_min_2h'] = df['wind_speed'].rolling(
            window=window_size, min_periods=12).min()

        df['wind_speed_max_2h'] = df['wind_speed'].rolling(
            window=window_size, min_periods=12).max()

        # Gust factor
        df['gust_factor'] = df['wind_gust'] / (df['wind_speed'] + 0.1)

        return df

    def _circular_range(self, angles):
        """Calculate the range of circular data (wind direction)"""
        if len(angles) < 2:
            return np.nan

        # Convert to radians
        angles_rad = np.radians(angles)

        # Calculate circular variance
        sin_sum = np.sum(np.sin(angles_rad))
        cos_sum = np.sum(np.cos(angles_rad))

        # Mean direction
        mean_angle = np.arctan2(sin_sum, cos_sum)

        # Calculate angular differences from mean
        diff = angles_rad - mean_angle
        diff = np.arctan2(np.sin(diff), np.cos(diff))

        # Return range in degrees
        return np.degrees(np.max(diff) - np.min(diff))

    def create_sailing_target(self, df):
        """Create target variable for good sailing conditions"""
        df = df.copy()

        # Filter for sailing hours (9AM to 4PM)
        sailing_hours = df.between_time('09:00', '16:00')

        # Initialize target variable
        df['good_sailing'] = 0

        # Criteria for good sailing conditions:
        # 1. Wind speed between 2 and 12 knots
        # 2. Wind direction consistency within 30 degrees over 2 hours
        # 3. Only during sailing hours (9AM-4PM)

        sailing_conditions = (
            (sailing_hours['wind_speed'] >= 2) &
            (sailing_hours['wind_speed'] <= 12) &
            (sailing_hours['wind_dir_range_2h'] <= 30) &
            (sailing_hours['wind_dir_range_2h'].notna())
        )

        df.loc[sailing_conditions.index[sailing_conditions], 'good_sailing'] = 1

        return df

    def create_features(self, df):
        """Create features for machine learning model"""
        df = df.copy()

        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

        # Seasonal features
        df['day_of_year'] = df.index.dayofyear
        df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        # Wind direction circular features
        df['wind_dir_sin'] = np.sin(np.radians(df['wind_direction']))
        df['wind_dir_cos'] = np.cos(np.radians(df['wind_direction']))

        # Pressure trend (change over last hour)
        df['pressure_trend_1h'] = df['pressure_relative'].diff(periods=12)

        # Temperature trend
        df['temp_trend_1h'] = df['temperature'].diff(periods=12)

        # Lag features (previous day same time)
        lag_24h = 24 * 12  # 24 hours in 5-minute intervals
        df['wind_speed_lag_24h'] = df['wind_speed'].shift(lag_24h)
        df['wind_dir_range_lag_24h'] = df['wind_dir_range_2h'].shift(lag_24h)

        return df

    def prepare_training_data(self, df):
        """Prepare data for machine learning"""
        # Feature columns
        feature_cols = [
            'temperature', 'feels_like', 'humidity', 'pressure_relative',
            'wind_speed', 'wind_gust', 'wind_dir_std_2h', 'wind_dir_range_2h',
            'wind_speed_mean_2h', 'wind_speed_std_2h', 'wind_speed_min_2h',
            'wind_speed_max_2h', 'gust_factor', 'hour', 'day_of_week',
            'month', 'is_weekend', 'sin_day_of_year', 'cos_day_of_year',
            'wind_dir_sin', 'wind_dir_cos', 'pressure_trend_1h',
            'temp_trend_1h', 'wind_speed_lag_24h', 'wind_dir_range_lag_24h'
        ]

        # Filter available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_cols

        # Remove rows with missing target or key features
        df_clean = df.dropna(subset=['good_sailing', 'wind_speed', 'wind_direction'])

        # Create feature matrix
        X = df_clean[available_cols].fillna(df_clean[available_cols].median())
        y = df_clean['good_sailing']

        return X, y, df_clean

    def train_model(self, X, y):
        """Train the sailing prediction model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 most important features:")
        print(feature_importance.head(10))

        # Classification report
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return X_train, X_test, y_train, y_test

    def predict_sailing_conditions(self, df):
        """Predict sailing conditions for new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")

        # Create features
        df_features = self.create_features(
            self.create_wind_consistency_features(df)
        )

        # Prepare feature matrix
        X = df_features[self.feature_columns].fillna(
            df_features[self.feature_columns].median()
        )

        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        # Add predictions to dataframe
        df_features['sailing_prediction'] = predictions
        df_features['sailing_probability'] = probabilities

        return df_features

    def analyze_sailing_patterns(self, df):
        """Analyze sailing patterns in the data"""
        plt.figure(figsize=(15, 10))

        # 1. Sailing conditions by hour
        plt.subplot(2, 3, 1)
        hourly_sailing = df.groupby(df.index.hour)['good_sailing'].mean()
        hourly_sailing.plot(kind='bar')
        plt.title('Good Sailing Conditions by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Proportion of Good Sailing')
        plt.xticks(rotation=45)

        # 2. Sailing conditions by month
        plt.subplot(2, 3, 2)
        monthly_sailing = df.groupby(df.index.month)['good_sailing'].mean()
        monthly_sailing.plot(kind='bar')
        plt.title('Good Sailing Conditions by Month')
        plt.xlabel('Month')
        plt.ylabel('Proportion of Good Sailing')

        # 3. Wind speed distribution
        plt.subplot(2, 3, 3)
        plt.hist(df[df['good_sailing'] == 1]['wind_speed'], alpha=0.7, label='Good Sailing', bins=30)
        plt.hist(df[df['good_sailing'] == 0]['wind_speed'], alpha=0.7, label='Poor Sailing', bins=30)
        plt.xlabel('Wind Speed (knots)')
        plt.ylabel('Frequency')
        plt.title('Wind Speed Distribution')
        plt.legend()

        # 4. Wind direction consistency
        plt.subplot(2, 3, 4)
        plt.hist(df[df['good_sailing'] == 1]['wind_dir_range_2h'], alpha=0.7, label='Good Sailing', bins=30)
        plt.hist(df[df['good_sailing'] == 0]['wind_dir_range_2h'], alpha=0.7, label='Poor Sailing', bins=30)
        plt.xlabel('Wind Direction Range (degrees)')
        plt.ylabel('Frequency')
        plt.title('Wind Direction Consistency')
        plt.legend()

        # 5. Correlation heatmap
        plt.subplot(2, 3, 5)
        corr_cols = ['wind_speed', 'wind_dir_range_2h', 'temperature', 'pressure_relative', 'good_sailing']
        corr_matrix = df[corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation')

        # 6. Seasonal pattern
        plt.subplot(2, 3, 6)
        df['date'] = df.index.date
        daily_sailing = df.groupby('date')['good_sailing'].mean()
        daily_sailing.plot()
        plt.title('Sailing Conditions Over Time')
        plt.xlabel('Date')
        plt.ylabel('Proportion of Good Sailing')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

        # Print statistics
        print(f"Overall good sailing conditions: {df['good_sailing'].mean():.3f}")
        print(f"Total sailing hours: {df['good_sailing'].sum()}")
        print(f"Total observations: {len(df)}")

def main():
    # Initialize predictor
    predictor = SailingWindPredictor()

    # Example usage
    print("Sailing Wind Prediction System")
    print("=" * 40)

    # Load data (replace with your actual file path)
    file_path = "input/Wetterstation July 16 2025.xlsx"  # Update this path to your Excel file

    try:
        # Load and process data
        df = predictor.load_excel_data(file_path)
        if df is not None:
            df = predictor.create_wind_consistency_features(df)
            df = predictor.create_sailing_target(df)
            df = predictor.create_features(df)

            # Analyze patterns
            predictor.analyze_sailing_patterns(df)

            # Prepare training data
            X, y, df_clean = predictor.prepare_training_data(df)

            # Train model
            X_train, X_test, y_train, y_test = predictor.train_model(X, y)

            # Save processed data
            df_clean.to_csv('processed_weather_data.csv')
            print("\nProcessed data saved to 'processed_weather_data.csv'")

    except FileNotFoundError:
        print(f"File {file_path} not found. Please update the file path.")
        print("\nTo use this system:")
        print("1. Create virtual environment: python3 -m venv sailing_env")
        print("2. Activate environment: source sailing_env/bin/activate")
        print("3. Install packages: pip install pandas numpy matplotlib seaborn scikit-learn openpyxl")
        print("4. Update the file_path variable with your Excel file path")
        print("5. Run the script to process your data and train the model")
        print("6. Use the trained model to predict sailing conditions")

if __name__ == "__main__":
    main()
