from flask import Flask, request, render_template
import cv2
import numpy as np
import os
import math
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
API_KEY = 'a05450939ec006a8e3d1a0ac3c494500'  # Replace with your OpenWeatherMap API key


class SolarPanelPlacement:
    def __init__(self, image_path):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not read image from {image_path}")
        self.panel_width = 1.0
        self.panel_height = 1.65
        self.panel_area = self.panel_width * self.panel_height
        self.panel_gap = 0.1
        self.sqm_to_sqft = 10.7639
        self.panel_efficiency = 0.3  # Assumes each panel generates 0.3 kW per panel
        self.sunlight_hours_per_day = 5  # Default sunlight hours if API fails

    def preprocess_image(self):
        max_width = 800
        if self.original_image.shape[1] > max_width:
            scale_factor = max_width / self.original_image.shape[1]
            self.original_image = cv2.resize(
                self.original_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA
            )
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.filtered_image = cv2.bilateralFilter(self.gray_image, 11, 17, 17)
        return self

    def detect_rooftops(self):
        _, binary = cv2.threshold(self.filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.rooftop_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 20]
        return self

    def estimate_panel_placement(self):
        panel_placement_results = {'total_rooftops': len(self.rooftop_contours), 'panel_placement_details': []}
        for i, contour in enumerate(self.rooftop_contours):
            x, y, w, h = cv2.boundingRect(contour)
            rooftop_width = w * 0.1
            rooftop_height = h * 0.1
            rooftop_area_sqm = rooftop_width * rooftop_height
            rooftop_area_sqft = rooftop_area_sqm * self.sqm_to_sqft
            panels_horizontally = max(1, math.floor(rooftop_width / (self.panel_width + self.panel_gap)))
            panels_vertically = max(1, math.floor(rooftop_height / (self.panel_height + self.panel_gap)))
            total_panels = panels_horizontally * panels_vertically
            estimated_capacity = total_panels * self.panel_efficiency  # in kW
            daily_capacity = estimated_capacity * self.sunlight_hours_per_day
            monthly_capacity = daily_capacity * 30
            yearly_capacity = daily_capacity * 365
            rooftop_details = {
                'rooftop_index': i + 1,
                'rooftop_area_m2': rooftop_area_sqm,
                'rooftop_area_sqft': rooftop_area_sqft,
                'panels_horizontally': panels_horizontally,
                'panels_vertically': panels_vertically,
                'total_panels': total_panels,
                'estimated_capacity_kW': estimated_capacity,
                'daily_capacity_kWh': daily_capacity,
                'monthly_capacity_kWh': monthly_capacity,
                'yearly_capacity_kWh': yearly_capacity
            }
            panel_placement_results['panel_placement_details'].append(rooftop_details)
        panel_placement_results['total_panels'] = sum(
            details['total_panels'] for details in panel_placement_results['panel_placement_details']
        )
        panel_placement_results['total_estimated_capacity_kW'] = sum(
            details['estimated_capacity_kW'] for details in panel_placement_results['panel_placement_details']
        )
        panel_placement_results['total_daily_capacity_kWh'] = sum(
            details['daily_capacity_kWh'] for details in panel_placement_results['panel_placement_details']
        )
        panel_placement_results['total_monthly_capacity_kWh'] = sum(
            details['monthly_capacity_kWh'] for details in panel_placement_results['panel_placement_details']
        )
        panel_placement_results['total_yearly_capacity_kWh'] = sum(
            details['yearly_capacity_kWh'] for details in panel_placement_results['panel_placement_details']
        )
        return panel_placement_results


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files or 'location' not in request.form:
            return "No file or location provided", 400

        file = request.files['image']
        location = request.form['location']

        if file.filename == '':
            return "No selected file", 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Get weather data
            weather_data = get_weather_data(location)
            if not weather_data:
                return "Could not retrieve weather data for the location", 400

            sunlight_hours = weather_data.get('sunlight_hours', 5)

            # Process the image and estimate panel placement
            assessment = SolarPanelPlacement(file_path)
            assessment.sunlight_hours_per_day = sunlight_hours
            assessment.preprocess_image().detect_rooftops()
            results = assessment.estimate_panel_placement()

            # Return the results to the template
            return render_template('results.html', results=results, location=location, weather=weather_data)
        except Exception as e:
            return f"Error: {e}", 500

    return render_template('index.html')


def get_weather_data(location):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        sunlight_hours = 5  # Default value
        if data['weather'][0]['main'].lower() in ['clear', 'few clouds']:
            sunlight_hours = 6
        elif data['weather'][0]['main'].lower() in ['clouds']:
            sunlight_hours = 4
        elif data['weather'][0]['main'].lower() in ['rain', 'snow']:
            sunlight_hours = 3
        return {
            'location': location,
            'temperature': data['main']['temp'],
            'weather_condition': data['weather'][0]['description'],
            'sunlight_hours': sunlight_hours,
        }
    return None


if __name__ == '__main__':
    app.run(debug=True)
