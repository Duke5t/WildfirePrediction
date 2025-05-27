# WildfirePrediction
(IN PROGRESS) ML Model used to predict size and spread of wildfires based on data available at point of ignition.


Aim:
  Using an amalgamation of historical data from Alberta, Canada wildfires 1983-1995, 1996-2005, 2006-2024, predict final size class of fire using only data from the initial ignition of the fire (ie. “FUEL_TYPE”, temperature, humidity, wind speed, wind direction, slope of the terrain). Use data to model a time series or heat map of fire spread from initial coordinate location.

Feature Selection:
  Quantitative
    “WIND_SPEED”
    “TEMPERATURE”
    “RELATIVE_HUMIDITY”
  Categorical
    “START_FOR_DATE_FIRE” vs “REPORT_DATE”??
    “WIND_DIRECTION”
    “FUEL_TYPE”
    “FIRE_POSITION_ON_SLOPE”
    “WEATHER_CONDITIONS_OVER_FIRE”


Preprocessing:

  
