import os
import pandas as pd

# Define the car dataset structure and data
cars_data = [
    # Maruti Suzuki
    {"Car Name": "Maruti Alto K10", "Company Name": "Maruti Suzuki", "Price": 3.99, "Mileage": 24.39, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 998},
    {"Car Name": "Maruti Alto K10 CNG", "Company Name": "Maruti Suzuki", "Price": 5.96, "Mileage": 33.85, "Fuel Type": "CNG", "Seats": 5, "Transmission": "Manual", "Engine": 998},
    {"Car Name": "Maruti S-Presso", "Company Name": "Maruti Suzuki", "Price": 4.26, "Mileage": 21.7, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 998},
    {"Car Name": "Maruti Celerio", "Company Name": "Maruti Suzuki", "Price": 5.37, "Mileage": 25.24, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 998},
    {"Car Name": "Maruti Celerio AMT", "Company Name": "Maruti Suzuki", "Price": 6.38, "Mileage": 26.68, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 998},
    {"Car Name": "Maruti Wagon R", "Company Name": "Maruti Suzuki", "Price": 5.54, "Mileage": 24.35, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1197},
    {"Car Name": "Maruti Wagon R CNG", "Company Name": "Maruti Suzuki", "Price": 6.89, "Mileage": 34.05, "Fuel Type": "CNG", "Seats": 5, "Transmission": "Manual", "Engine": 998},
    {"Car Name": "Maruti Swift", "Company Name": "Maruti Suzuki", "Price": 6.49, "Mileage": 25.75, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1197},
    {"Car Name": "Maruti Swift AMT", "Company Name": "Maruti Suzuki", "Price": 7.50, "Mileage": 25.75, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1197},
    {"Car Name": "Maruti Dzire", "Company Name": "Maruti Suzuki", "Price": 6.57, "Mileage": 22.41, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1197},
    {"Car Name": "Maruti Dzire AMT", "Company Name": "Maruti Suzuki", "Price": 7.90, "Mileage": 22.61, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1197},
    {"Car Name": "Maruti Baleno", "Company Name": "Maruti Suzuki", "Price": 6.66, "Mileage": 22.35, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1197},
    {"Car Name": "Maruti Brezza", "Company Name": "Maruti Suzuki", "Price": 8.34, "Mileage": 17.38, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1462},
    {"Car Name": "Maruti Brezza AT", "Company Name": "Maruti Suzuki", "Price": 11.10, "Mileage": 19.89, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1462},
    {"Car Name": "Maruti Ertiga", "Company Name": "Maruti Suzuki", "Price": 8.69, "Mileage": 20.51, "Fuel Type": "Petrol", "Seats": 7, "Transmission": "Manual", "Engine": 1462},
    {"Car Name": "Maruti Ertiga CNG", "Company Name": "Maruti Suzuki", "Price": 10.78, "Mileage": 26.11, "Fuel Type": "CNG", "Seats": 7, "Transmission": "Manual", "Engine": 1462},
    {"Car Name": "Maruti XL6", "Company Name": "Maruti Suzuki", "Price": 11.61, "Mileage": 20.97, "Fuel Type": "Petrol", "Seats": 6, "Transmission": "Manual", "Engine": 1462},
    {"Car Name": "Maruti Grand Vitara Hybrid", "Company Name": "Maruti Suzuki", "Price": 18.43, "Mileage": 27.97, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1490},
    {"Car Name": "Maruti Fronx", "Company Name": "Maruti Suzuki", "Price": 7.51, "Mileage": 21.79, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1197},
    {"Car Name": "Maruti Jimny", "Company Name": "Maruti Suzuki", "Price": 12.74, "Mileage": 16.94, "Fuel Type": "Petrol", "Seats": 4, "Transmission": "Manual", "Engine": 1462},

    # Hyundai
    {"Car Name": "Hyundai Grand i10 Nios", "Company Name": "Hyundai", "Price": 5.92, "Mileage": 20.7, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1197},
    {"Car Name": "Hyundai Grand i10 CNG", "Company Name": "Hyundai", "Price": 7.68, "Mileage": 27.3, "Fuel Type": "CNG", "Seats": 5, "Transmission": "Manual", "Engine": 1197},
    {"Car Name": "Hyundai i20", "Company Name": "Hyundai", "Price": 7.04, "Mileage": 20.3, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1197},
    {"Car Name": "Hyundai i20 N Line", "Company Name": "Hyundai", "Price": 9.99, "Mileage": 19.6, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 998},
    {"Car Name": "Hyundai Aura", "Company Name": "Hyundai", "Price": 6.49, "Mileage": 20.5, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1197},
    {"Car Name": "Hyundai Exter", "Company Name": "Hyundai", "Price": 6.13, "Mileage": 19.4, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1197},
    {"Car Name": "Hyundai Venue", "Company Name": "Hyundai", "Price": 7.94, "Mileage": 17.5, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1197},
    {"Car Name": "Hyundai Venue Turbo DCT", "Company Name": "Hyundai", "Price": 11.86, "Mileage": 18.3, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 998},
    {"Car Name": "Hyundai Creta Petrol", "Company Name": "Hyundai", "Price": 11.00, "Mileage": 17.4, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1497},
    {"Car Name": "Hyundai Creta Diesel", "Company Name": "Hyundai", "Price": 12.45, "Mileage": 21.8, "Fuel Type": "Diesel", "Seats": 5, "Transmission": "Manual", "Engine": 1493},
    {"Car Name": "Hyundai Creta IVT", "Company Name": "Hyundai", "Price": 15.82, "Mileage": 17.7, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1497},
    {"Car Name": "Hyundai Alcazar Diesel", "Company Name": "Hyundai", "Price": 17.78, "Mileage": 20.4, "Fuel Type": "Diesel", "Seats": 7, "Transmission": "Manual", "Engine": 1493},
    {"Car Name": "Hyundai Verna Turbo", "Company Name": "Hyundai", "Price": 14.84, "Mileage": 20.0, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1482},
    {"Car Name": "Hyundai Verna DCT", "Company Name": "Hyundai", "Price": 16.08, "Mileage": 20.6, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1482},
    {"Car Name": "Hyundai Tucson", "Company Name": "Hyundai", "Price": 29.02, "Mileage": 18.0, "Fuel Type": "Diesel", "Seats": 5, "Transmission": "Automatic", "Engine": 1997},
    {"Car Name": "Hyundai Ioniq 5 EV", "Company Name": "Hyundai", "Price": 46.05, "Mileage": 31.0, "Fuel Type": "Electric", "Seats": 5, "Transmission": "Automatic", "Engine": 0},

    # Tata Motors
    {"Car Name": "Tata Tiago", "Company Name": "Tata Motors", "Price": 5.65, "Mileage": 20.01, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1199},
    {"Car Name": "Tata Tiago CNG", "Company Name": "Tata Motors", "Price": 6.55, "Mileage": 26.49, "Fuel Type": "CNG", "Seats": 5, "Transmission": "Manual", "Engine": 1199},
    {"Car Name": "Tata Tiago EV", "Company Name": "Tata Motors", "Price": 7.99, "Mileage": 28.0, "Fuel Type": "Electric", "Seats": 5, "Transmission": "Automatic", "Engine": 0},
    {"Car Name": "Tata Tigor", "Company Name": "Tata Motors", "Price": 6.30, "Mileage": 19.28, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1199},
    {"Car Name": "Tata Punch", "Company Name": "Tata Motors", "Price": 6.13, "Mileage": 20.09, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1199},
    {"Car Name": "Tata Punch EV", "Company Name": "Tata Motors", "Price": 10.99, "Mileage": 29.5, "Fuel Type": "Electric", "Seats": 5, "Transmission": "Automatic", "Engine": 0},
    {"Car Name": "Tata Altroz Petrol", "Company Name": "Tata Motors", "Price": 6.60, "Mileage": 19.33, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1199},
    {"Car Name": "Tata Altroz Diesel", "Company Name": "Tata Motors", "Price": 8.80, "Mileage": 23.64, "Fuel Type": "Diesel", "Seats": 5, "Transmission": "Manual", "Engine": 1497},
    {"Car Name": "Tata Nexon Facelift", "Company Name": "Tata Motors", "Price": 8.15, "Mileage": 17.44, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1199},
    {"Car Name": "Tata Nexon Diesel", "Company Name": "Tata Motors", "Price": 11.10, "Mileage": 23.23, "Fuel Type": "Diesel", "Seats": 5, "Transmission": "Manual", "Engine": 1497},
    {"Car Name": "Tata Nexon EV", "Company Name": "Tata Motors", "Price": 14.49, "Mileage": 30.5, "Fuel Type": "Electric", "Seats": 5, "Transmission": "Automatic", "Engine": 0},
    {"Car Name": "Tata Harrier", "Company Name": "Tata Motors", "Price": 15.49, "Mileage": 16.8, "Fuel Type": "Diesel", "Seats": 5, "Transmission": "Manual", "Engine": 1956},
    {"Car Name": "Tata Harrier Automatic", "Company Name": "Tata Motors", "Price": 19.99, "Mileage": 14.6, "Fuel Type": "Diesel", "Seats": 5, "Transmission": "Automatic", "Engine": 1956},
    {"Car Name": "Tata Safari", "Company Name": "Tata Motors", "Price": 16.19, "Mileage": 16.3, "Fuel Type": "Diesel", "Seats": 7, "Transmission": "Manual", "Engine": 1956},
    {"Car Name": "Tata Safari Automatic", "Company Name": "Tata Motors", "Price": 20.69, "Mileage": 14.5, "Fuel Type": "Diesel", "Seats": 7, "Transmission": "Automatic", "Engine": 1956},
    {"Car Name": "Tata Curvv EV", "Company Name": "Tata Motors", "Price": 17.49, "Mileage": 33.0, "Fuel Type": "Electric", "Seats": 5, "Transmission": "Automatic", "Engine": 0},

    # Honda
    {"Car Name": "Honda Amaze", "Company Name": "Honda", "Price": 7.16, "Mileage": 18.6, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1199},
    {"Car Name": "Honda Amaze CVT", "Company Name": "Honda", "Price": 8.78, "Mileage": 18.3, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1199},
    {"Car Name": "Honda City", "Company Name": "Honda", "Price": 11.71, "Mileage": 17.8, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1498},
    {"Car Name": "Honda City CVT", "Company Name": "Honda", "Price": 13.84, "Mileage": 18.4, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1498},
    {"Car Name": "Honda City Hybrid e:HEV", "Company Name": "Honda", "Price": 18.89, "Mileage": 27.1, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1498},
    {"Car Name": "Honda Elevate", "Company Name": "Honda", "Price": 11.58, "Mileage": 15.31, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1498},
    {"Car Name": "Honda Elevate CVT", "Company Name": "Honda", "Price": 13.21, "Mileage": 16.92, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1498},

    # Mahindra
    {"Car Name": "Mahindra Thar RWD", "Company Name": "Mahindra", "Price": 11.25, "Mileage": 15.2, "Fuel Type": "Diesel", "Seats": 4, "Transmission": "Manual", "Engine": 1497},
    {"Car Name": "Mahindra Thar 4WD Petrol", "Company Name": "Mahindra", "Price": 14.30, "Mileage": 12.8, "Fuel Type": "Petrol", "Seats": 4, "Transmission": "Manual", "Engine": 1997},
    {"Car Name": "Mahindra Thar 4WD Diesel", "Company Name": "Mahindra", "Price": 14.85, "Mileage": 13.0, "Fuel Type": "Diesel", "Seats": 4, "Transmission": "Manual", "Engine": 2184},
    {"Car Name": "Mahindra Scorpio Classic", "Company Name": "Mahindra", "Price": 13.59, "Mileage": 14.4, "Fuel Type": "Diesel", "Seats": 7, "Transmission": "Manual", "Engine": 2179},
    {"Car Name": "Mahindra Scorpio-N Petrol", "Company Name": "Mahindra", "Price": 13.60, "Mileage": 12.5, "Fuel Type": "Petrol", "Seats": 7, "Transmission": "Manual", "Engine": 1997},
    {"Car Name": "Mahindra Scorpio-N Diesel", "Company Name": "Mahindra", "Price": 14.25, "Mileage": 14.0, "Fuel Type": "Diesel", "Seats": 7, "Transmission": "Manual", "Engine": 2184},
    {"Car Name": "Mahindra Scorpio-N AT", "Company Name": "Mahindra", "Price": 16.99, "Mileage": 13.5, "Fuel Type": "Diesel", "Seats": 7, "Transmission": "Automatic", "Engine": 2184},
    {"Car Name": "Mahindra XUV300 Petrol", "Company Name": "Mahindra", "Price": 7.99, "Mileage": 17.0, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1197},
    {"Car Name": "Mahindra XUV300 Diesel", "Company Name": "Mahindra", "Price": 9.90, "Mileage": 20.0, "Fuel Type": "Diesel", "Seats": 5, "Transmission": "Manual", "Engine": 1497},
    {"Car Name": "Mahindra XUV400 EV", "Company Name": "Mahindra", "Price": 15.49, "Mileage": 31.2, "Fuel Type": "Electric", "Seats": 5, "Transmission": "Automatic", "Engine": 0},
    {"Car Name": "Mahindra XUV700 Petrol AX5", "Company Name": "Mahindra", "Price": 17.69, "Mileage": 13.0, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1997},
    {"Car Name": "Mahindra XUV700 Diesel AX7", "Company Name": "Mahindra", "Price": 19.79, "Mileage": 15.5, "Fuel Type": "Diesel", "Seats": 7, "Transmission": "Manual", "Engine": 2184},
    {"Car Name": "Mahindra XUV700 Diesel AT", "Company Name": "Mahindra", "Price": 21.59, "Mileage": 14.8, "Fuel Type": "Diesel", "Seats": 7, "Transmission": "Automatic", "Engine": 2184},
    {"Car Name": "Mahindra Bolero Neo", "Company Name": "Mahindra", "Price": 9.90, "Mileage": 17.2, "Fuel Type": "Diesel", "Seats": 7, "Transmission": "Manual", "Engine": 1493},

    # Toyota
    {"Car Name": "Toyota Glanza", "Company Name": "Toyota", "Price": 6.81, "Mileage": 22.35, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1197},
    {"Car Name": "Toyota Urban Cruiser Taisor", "Company Name": "Toyota", "Price": 7.74, "Mileage": 21.5, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1197},
    {"Car Name": "Toyota Urban Cruiser Hyryder", "Company Name": "Toyota", "Price": 11.14, "Mileage": 21.11, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1462},
    {"Car Name": "Toyota Hyryder Hybrid", "Company Name": "Toyota", "Price": 16.66, "Mileage": 27.97, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1490},
    {"Car Name": "Toyota Innova Crysta Diesel", "Company Name": "Toyota", "Price": 19.99, "Mileage": 12.0, "Fuel Type": "Diesel", "Seats": 7, "Transmission": "Manual", "Engine": 2393},
    {"Car Name": "Toyota Innova Hycross Hybrid", "Company Name": "Toyota", "Price": 25.97, "Mileage": 23.24, "Fuel Type": "Petrol", "Seats": 7, "Transmission": "Automatic", "Engine": 1987},
    {"Car Name": "Toyota Fortuner Petrol", "Company Name": "Toyota", "Price": 33.43, "Mileage": 10.0, "Fuel Type": "Petrol", "Seats": 7, "Transmission": "Manual", "Engine": 2694},
    {"Car Name": "Toyota Fortuner Diesel", "Company Name": "Toyota", "Price": 35.93, "Mileage": 12.5, "Fuel Type": "Diesel", "Seats": 7, "Transmission": "Manual", "Engine": 2755},
    {"Car Name": "Toyota Fortuner Diesel AT", "Company Name": "Toyota", "Price": 38.21, "Mileage": 11.8, "Fuel Type": "Diesel", "Seats": 7, "Transmission": "Automatic", "Engine": 2755},
    {"Car Name": "Toyota Hilux", "Company Name": "Toyota", "Price": 30.40, "Mileage": 12.0, "Fuel Type": "Diesel", "Seats": 5, "Transmission": "Manual", "Engine": 2755},
    {"Car Name": "Toyota Camry Hybrid", "Company Name": "Toyota", "Price": 46.17, "Mileage": 22.7, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 2487},

    # Kia
    {"Car Name": "Kia Sonet Petrol", "Company Name": "Kia", "Price": 7.99, "Mileage": 18.8, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1197},
    {"Car Name": "Kia Sonet Turbo DCT", "Company Name": "Kia", "Price": 11.99, "Mileage": 19.2, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 998},
    {"Car Name": "Kia Sonet Diesel", "Company Name": "Kia", "Price": 9.79, "Mileage": 24.1, "Fuel Type": "Diesel", "Seats": 5, "Transmission": "Manual", "Engine": 1493},
    {"Car Name": "Kia Seltos Petrol", "Company Name": "Kia", "Price": 10.90, "Mileage": 17.0, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1497},
    {"Car Name": "Kia Seltos Diesel", "Company Name": "Kia", "Price": 12.00, "Mileage": 20.7, "Fuel Type": "Diesel", "Seats": 5, "Transmission": "Manual", "Engine": 1493},
    {"Car Name": "Kia Seltos Turbo DCT", "Company Name": "Kia", "Price": 18.30, "Mileage": 17.9, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1482},
    {"Car Name": "Kia Carens Petrol", "Company Name": "Kia", "Price": 10.45, "Mileage": 17.9, "Fuel Type": "Petrol", "Seats": 7, "Transmission": "Manual", "Engine": 1497},
    {"Car Name": "Kia Carens Diesel AT", "Company Name": "Kia", "Price": 16.55, "Mileage": 18.4, "Fuel Type": "Diesel", "Seats": 7, "Transmission": "Automatic", "Engine": 1493},
    {"Car Name": "Kia EV6", "Company Name": "Kia", "Price": 60.95, "Mileage": 32.5, "Fuel Type": "Electric", "Seats": 5, "Transmission": "Automatic", "Engine": 0},

    # Skoda & Volkswagen
    {"Car Name": "Skoda Slavia 1.0 TSI", "Company Name": "Skoda", "Price": 11.53, "Mileage": 19.47, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 999},
    {"Car Name": "Skoda Slavia 1.5 TSI DSG", "Company Name": "Skoda", "Price": 16.63, "Mileage": 18.73, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1498},
    {"Car Name": "Skoda Kushaq 1.0 TSI", "Company Name": "Skoda", "Price": 11.89, "Mileage": 17.88, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 999},
    {"Car Name": "Skoda Kushaq 1.5 TSI DSG", "Company Name": "Skoda", "Price": 17.39, "Mileage": 18.09, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1498},
    {"Car Name": "Volkswagen Virtus 1.0 TSI", "Company Name": "Volkswagen", "Price": 11.48, "Mileage": 19.4, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 999},
    {"Car Name": "Volkswagen Virtus 1.5 GT DSG", "Company Name": "Volkswagen", "Price": 16.19, "Mileage": 18.6, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1498},
    {"Car Name": "Volkswagen Taigun 1.0 TSI", "Company Name": "Volkswagen", "Price": 11.62, "Mileage": 17.8, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 999},
    {"Car Name": "Volkswagen Taigun 1.5 GT DCT", "Company Name": "Volkswagen", "Price": 16.79, "Mileage": 18.1, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1498},

    # MG & BYD
    {"Car Name": "MG Astor Petrol", "Company Name": "MG Motors", "Price": 9.98, "Mileage": 15.4, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Manual", "Engine": 1498},
    {"Car Name": "MG Hector Diesel", "Company Name": "MG Motors", "Price": 18.28, "Mileage": 15.5, "Fuel Type": "Diesel", "Seats": 5, "Transmission": "Manual", "Engine": 1956},
    {"Car Name": "MG ZS EV", "Company Name": "MG Motors", "Price": 18.98, "Mileage": 30.0, "Fuel Type": "Electric", "Seats": 5, "Transmission": "Automatic", "Engine": 0},
    {"Car Name": "MG Comet EV", "Company Name": "MG Motors", "Price": 6.99, "Mileage": 23.0, "Fuel Type": "Electric", "Seats": 4, "Transmission": "Automatic", "Engine": 0},
    {"Car Name": "BYD Atto 3 EV", "Company Name": "BYD", "Price": 33.99, "Mileage": 32.1, "Fuel Type": "Electric", "Seats": 5, "Transmission": "Automatic", "Engine": 0},
    {"Car Name": "BYD Seal EV Premium", "Company Name": "BYD", "Price": 45.55, "Mileage": 36.0, "Fuel Type": "Electric", "Seats": 5, "Transmission": "Automatic", "Engine": 0},

    # Premium Luxury (BMW, Audi, Mercedes)
    {"Car Name": "BMW 3 Series Gran Limousine", "Company Name": "BMW", "Price": 60.60, "Mileage": 15.39, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1998},
    {"Car Name": "BMW 3 Series GL Diesel", "Company Name": "BMW", "Price": 62.50, "Mileage": 19.61, "Fuel Type": "Diesel", "Seats": 5, "Transmission": "Automatic", "Engine": 1995},
    {"Car Name": "BMW X1 sDrive20i", "Company Name": "BMW", "Price": 49.50, "Mileage": 16.35, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1499},
    {"Car Name": "BMW X1 sDrive18d", "Company Name": "BMW", "Price": 52.50, "Mileage": 20.37, "Fuel Type": "Diesel", "Seats": 5, "Transmission": "Automatic", "Engine": 1995},
    {"Car Name": "BMW X5 xDrive40i", "Company Name": "BMW", "Price": 96.00, "Mileage": 12.0, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 2998},
    {"Car Name": "BMW X5 xDrive30d", "Company Name": "BMW", "Price": 98.50, "Mileage": 12.0, "Fuel Type": "Diesel", "Seats": 5, "Transmission": "Automatic", "Engine": 2993},
    {"Car Name": "BMW i4 eDrive40 EV", "Company Name": "BMW", "Price": 72.50, "Mileage": 35.0, "Fuel Type": "Electric", "Seats": 5, "Transmission": "Automatic", "Engine": 0},
    
    {"Car Name": "Audi A4 40 TFSI", "Company Name": "Audi", "Price": 45.34, "Mileage": 17.4, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1984},
    {"Car Name": "Audi Q3 40 TFSI", "Company Name": "Audi", "Price": 43.80, "Mileage": 14.93, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1984},
    {"Car Name": "Audi Q5 45 TFSI", "Company Name": "Audi", "Price": 65.18, "Mileage": 13.4, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1984},
    {"Car Name": "Audi A6 45 TFSI", "Company Name": "Audi", "Price": 64.20, "Mileage": 14.11, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1984},
    {"Car Name": "Audi Q7 55 TFSI", "Company Name": "Audi", "Price": 88.66, "Mileage": 11.2, "Fuel Type": "Petrol", "Seats": 7, "Transmission": "Automatic", "Engine": 2995},
    {"Car Name": "Audi e-tron 55 EV", "Company Name": "Audi", "Price": 124.00, "Mileage": 30.0, "Fuel Type": "Electric", "Seats": 5, "Transmission": "Automatic", "Engine": 0},
    
    {"Car Name": "Mercedes-Benz A-Class Limousine", "Company Name": "Mercedes-Benz", "Price": 46.05, "Mileage": 17.5, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1332},
    {"Car Name": "Mercedes-Benz C-Class C200", "Company Name": "Mercedes-Benz", "Price": 61.85, "Mileage": 16.9, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1496},
    {"Car Name": "Mercedes-Benz C-Class C220d", "Company Name": "Mercedes-Benz", "Price": 63.00, "Mileage": 23.0, "Fuel Type": "Diesel", "Seats": 5, "Transmission": "Automatic", "Engine": 1993},
    {"Car Name": "Mercedes-Benz E-Class E200", "Company Name": "Mercedes-Benz", "Price": 76.05, "Mileage": 15.0, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1991},
    {"Car Name": "Mercedes-Benz E-Class E220d", "Company Name": "Mercedes-Benz", "Price": 77.25, "Mileage": 16.1, "Fuel Type": "Diesel", "Seats": 5, "Transmission": "Automatic", "Engine": 1993},
    {"Car Name": "Mercedes-Benz GLA 200", "Company Name": "Mercedes-Benz", "Price": 51.75, "Mileage": 17.4, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1332},
    {"Car Name": "Mercedes-Benz GLC 300", "Company Name": "Mercedes-Benz", "Price": 74.45, "Mileage": 12.74, "Fuel Type": "Petrol", "Seats": 5, "Transmission": "Automatic", "Engine": 1999},
    {"Car Name": "Mercedes-Benz GLE 300d", "Company Name": "Mercedes-Benz", "Price": 96.65, "Mileage": 14.0, "Fuel Type": "Diesel", "Seats": 5, "Transmission": "Automatic", "Engine": 1993},
    {"Car Name": "Mercedes-Benz GLS 450d", "Company Name": "Mercedes-Benz", "Price": 137.00, "Mileage": 11.5, "Fuel Type": "Diesel", "Seats": 7, "Transmission": "Automatic", "Engine": 2989},
    {"Car Name": "Mercedes-Benz EQE 500 SUV EV", "Company Name": "Mercedes-Benz", "Price": 139.00, "Mileage": 34.5, "Fuel Type": "Electric", "Seats": 5, "Transmission": "Automatic", "Engine": 0},

    # Tesla & Luxury EVs
    {"Car Name": "Tesla Model 3 Standard", "Company Name": "Tesla", "Price": 45.00, "Mileage": 32.0, "Fuel Type": "Electric", "Seats": 5, "Transmission": "Automatic", "Engine": 0},
    {"Car Name": "Tesla Model Y Long Range", "Company Name": "Tesla", "Price": 55.00, "Mileage": 33.5, "Fuel Type": "Electric", "Seats": 5, "Transmission": "Automatic", "Engine": 0},
    {"Car Name": "Tesla Model S Plaid", "Company Name": "Tesla", "Price": 110.00, "Mileage": 36.5, "Fuel Type": "Electric", "Seats": 5, "Transmission": "Automatic", "Engine": 0},
    {"Car Name": "Tesla Model X Dual Motor", "Company Name": "Tesla", "Price": 120.00, "Mileage": 34.0, "Fuel Type": "Electric", "Seats": 7, "Transmission": "Automatic", "Engine": 0}
]

# Add more variations to reach a solid sample count (currently 100+ unique detailed cars, we'll double check)
# Let's populate any additional entries to reach a robust dataset size and varied parameters
print(f"Loaded {len(cars_data)} unique car profiles. Generating dataset...")

# Create DataFrame
df = pd.DataFrame(cars_data)

# Ensure the data directory exists relative to this script's directory
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(DIR_PATH, "data")
os.makedirs(data_dir, exist_ok=True)

# Save to CSV
output_path = os.path.join(data_dir, "cars.csv")
df.to_csv(output_path, index=False)
print(f"Dataset successfully created and saved to {output_path}!")
print(df.head())
print("\nDataset Summary Stats:")
print(df.describe(include='all'))
