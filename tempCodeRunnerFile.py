from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import os
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Dataset path
dataset_path = 'sales_dataset_2023.csv'

# Load or initialize the dataset
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
else:
    df = pd.DataFrame(columns=['Year', 'Month', 'Product', 'Base Sales', 'Volume'])
    df.to_csv(dataset_path, index=False)

# Month order for validation
MONTH_ORDER = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]


@app.route('/')
def home():
    """Home page with input form."""
    brands = sorted(df['Product'].dropna().unique())
    volumes = sorted(df['Volume'].dropna().unique())
    return render_template('home.html', brands=brands, volumes=volumes)


@app.route('/predict', methods=['POST'])
def predict():
    """Predict sales for the specified month and year and show sales trend."""
    try:
        product = request.form.get('product')
        volume = request.form.get('volume')
        month = request.form.get('month')
        year = request.form.get('year')

        if not all([product, volume, month, year]):
            return render_template('result.html', prediction="All fields are required.")

        try:
            year = int(year)
        except ValueError:
            return render_template('result.html', prediction="Year must be a valid number.")

        if month not in MONTH_ORDER:
            return render_template('result.html', prediction=f"Invalid month '{month}'.")

        month_index = MONTH_ORDER.index(month)

        if year == 2024:
            # Determine previous months based on the target month
            if month == 'January':
                previous_months = ['October', 'November', 'December']
                check_years = [2023, 2023, 2023]
            elif month == 'February':
                previous_months = ['November', 'December', 'January']
                check_years = [2023, 2023, 2024]
            else:
                previous_months = [
                    MONTH_ORDER[month_index - 1],
                    MONTH_ORDER[month_index - 2],
                    MONTH_ORDER[month_index - 3]
                ]
                check_years = [year, year, year - 1 if month_index == 2 else year]

            # Check if any required data is missing
            missing_months = []
            for i, m in enumerate(previous_months):
                if df[
                    (df['Product'] == product) &
                    (df['Volume'] == volume) &
                    (df['Month'] == m) &
                    (df['Year'] == check_years[i])
                ].empty:
                    missing_months.append(m)

            if missing_months:
                # Redirect to add data for the first missing month
                return redirect(url_for('add_data', product=product, volume=volume, month=missing_months[0]))

            # Filter the data for the product and volume
            filtered_data = df[
                (df['Product'] == product) & 
                (df['Volume'] == volume)
            ].sort_values(by=['Year', 'Month'], key=lambda col: col.map(MONTH_ORDER.index))

            # Generate the sales trend chart
            img = io.BytesIO()
            plt.figure(figsize=(10, 6))
            plt.plot(
                filtered_data['Year'].astype(str) + ' ' + filtered_data['Month'],
                filtered_data['Base Sales'],
                marker='o',
                label=f'{product} - {volume}'
            )
            plt.xticks(rotation=45, ha='right')
            plt.title('Sales Trend (2023 Onwards)')
            plt.xlabel('Time')
            plt.ylabel('Base Sales')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            chart_url = base64.b64encode(img.getvalue()).decode()

            # Calculate the average sales
            recent_data = df[
                (df['Product'] == product) &
                (df['Volume'] == volume) &
                (df['Month'].isin(previous_months)) &
                (df['Year'].isin(check_years))
            ]
            average_sales = recent_data['Base Sales'].mean()

            return render_template(
                'result.html',
                prediction=f"Predicted sales for {product} ({volume}) in {month} 2024: {average_sales:.2f}.",
                chart_url=chart_url
            )

        else:
            return render_template(
                'result.html',
                prediction="Prediction is only available for 2024."
            )

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")


@app.route('/add_data')
def add_data():
    """Render form to add missing sales data."""
    product = request.args.get('product', '')
    volume = request.args.get('volume', '')
    month = request.args.get('month', '')

    brands = sorted(df['Product'].dropna().unique())
    volumes = sorted(df['Volume'].dropna().unique())

    return render_template('add_data.html', product=product, volume=volume, month=month, brands=brands, volumes=volumes)


@app.route('/submit_data', methods=['POST'])
def submit_data():
    """Handle submission of new sales data."""
    try:
        product = request.form.get('product')
        volume = request.form.get('volume')
        month = request.form.get('month')
        base_sales = request.form.get('base_sales')

        if not all([product, volume, month, base_sales]):
            return render_template('result.html', prediction="All fields are required.")

        try:
            base_sales = float(base_sales)
        except ValueError:
            return render_template('result.html', prediction="Base Sales must be a valid number.")

        if month not in MONTH_ORDER:
            return render_template('result.html', prediction=f"Invalid month '{month}'.")

        # Append the new data
        new_row = {
            'Year': 2024 if month != 'December' else 2023,
            'Month': month,
            'Product': product,
            'Base Sales': base_sales,
            'Volume': volume
        }
        global df
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(dataset_path, index=False)

        return render_template(
            'result.html',
            prediction=f"Sales data for {month} 2024 added successfully. You can now proceed with the prediction."
        )

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
