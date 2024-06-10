import math

def calculate_transit_depth(baseline_brightness, min_brightness, baseline_uncertainty, min_uncertainty):
    # Normalize the minimum brightness
    normalized_brightness = min_brightness / baseline_brightness
    
    # Calculate the transit depth
    transit_depth = 1 - normalized_brightness
    
    # Propagate the uncertainties
    normalized_uncertainty = normalized_brightness * math.sqrt(
        (min_uncertainty / min_brightness) ** 2 + 
        (baseline_uncertainty / baseline_brightness) ** 2
    )
    transit_depth_uncertainty = normalized_uncertainty
    
    return transit_depth, transit_depth_uncertainty

def calculate_planet_radius(transit_depth, star_radius_solar):
    # Convert star radius from solar radii to kilometers
    star_radius_km = star_radius_solar * 696340
    
    # Calculate the radius of the planet in kilometers
    planet_radius_km = star_radius_km * math.sqrt(transit_depth)
    
    # Convert planet radius from kilometers to Earth radii
    planet_radius_earth = planet_radius_km / 6371
    
    return planet_radius_earth

def calculate_planet_radius_uncertainty(transit_depth, transit_depth_uncertainty, star_radius_solar):
    # Convert star radius from solar radii to kilometers
    star_radius_km = star_radius_solar * 696340
    
    # Calculate the uncertainty in planet radius in kilometers
    planet_radius_uncertainty_km = star_radius_km * 0.5 * (transit_depth_uncertainty / math.sqrt(transit_depth))
    
    # Convert planet radius uncertainty from kilometers to Earth radii
    planet_radius_uncertainty_earth = planet_radius_uncertainty_km / 6371
    
    return planet_radius_uncertainty_earth

def main(baseline_brightness, min_brightness, baseline_uncertainty, min_uncertainty, star_radius_solar):
    # Calculate the transit depth and its uncertainty
    transit_depth, transit_depth_uncertainty = calculate_transit_depth(
        baseline_brightness, min_brightness, baseline_uncertainty, min_uncertainty
    )
    
    # Calculate the radius of the planet and its uncertainty in Earth radii
    planet_radius_earth = calculate_planet_radius(transit_depth, star_radius_solar)
    planet_radius_uncertainty_earth = calculate_planet_radius_uncertainty(transit_depth, transit_depth_uncertainty, star_radius_solar)
    
    return planet_radius_earth, planet_radius_uncertainty_earth

user_input = input("Enter baseline brightness, minimal brightness, baseline uncertainty, minimal uncertainty, and stellar radius in solar radii: ")
params = user_input.split()

baseline_brightness = float(params[0])
min_brightness = float(params[1])
baseline_uncertainty = float(params[2])
min_uncertainty = float(params[3])
star_radius_solar = float(params[4])

planet_radius, planet_radius_uncertainty = main(baseline_brightness, min_brightness, baseline_uncertainty, min_uncertainty, star_radius_solar)

print(f"Planet Radius: {planet_radius:.4f} Earth radii ± {planet_radius_uncertainty:.4f} Earth radii")