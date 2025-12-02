from geographic_scoring import score_location

test_locations = [
    (39.9526, -75.1652),
    (39.9612, -75.1598),
    (40.0018, -75.1338),
]

for lat, lon in test_locations:
    scores = score_location(lat, lon)
    print(f"Location: ({lat}, {lon})")
    print(f"Environmental Score: {scores['environmental_score']}/10")
    print(f"Recreational Score: {scores['recreational_score']}/10")
    print(f"Green Space Score: {scores['green_space_score']}/10")
    print(f"Walkability Score: {scores['walkability_score']}/10")
    print()

