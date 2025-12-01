"""
Demand Model for Land Use Recommendations

This module analyzes census demographic data to generate recommendations
for what to do with plots of land based on community needs.

PUBLIC SPACE RECOMMENDATIONS:
- Areas with many families and young children → recommend parks/playgrounds
- Areas with many seniors → recommend senior centers, walking trails
- Areas with many transit commuters → recommend transit-oriented development
- Areas with families → recommend community gardens

HOUSING RECOMMENDATIONS:
- Low-income areas with high rent burden → recommend affordable housing
- Areas with many families → recommend family housing
- Areas with many seniors → recommend senior housing
- Areas near transit → recommend transit-oriented housing
- Diverse moderate-income areas → recommend mixed-income housing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os


# Recommendation categories
PUBLIC_SPACE_TYPES = [
    "park", "community_garden", "senior_center", "walking_trail", 
    "transit_hub", "sports_facility", "urban_farm", "plaza"
]

HOUSING_TYPES = [
    "affordable_housing", "mixed_income_housing", "senior_housing",
    "family_housing", "transit_oriented_housing"
]

# Land use recommendation types
RECOMMENDATION_TYPES = {
    "park": {
        "name": "Park/Playground",
        "description": "Recreational park with playground equipment for children and families",
        "icon": "🌳"
    },
    "community_garden": {
        "name": "Community Garden",
        "description": "Shared garden space for community members to grow food and plants",
        "icon": "🌱"
    },
    "senior_center": {
        "name": "Senior Center/Recreation",
        "description": "Recreation center and facilities for senior citizens",
        "icon": "👴"
    },
    "walking_trail": {
        "name": "Walking Trail/Pathway",
        "description": "Paved or natural trail for walking, jogging, and biking",
        "icon": "🚶"
    },
    "transit_hub": {
        "name": "Transit-Oriented Development",
        "description": "Mixed-use development near transit for commuters",
        "icon": "🚇"
    },
    "sports_facility": {
        "name": "Sports Facility",
        "description": "Athletic fields, courts, or sports complex",
        "icon": "⚽"
    },
    "urban_farm": {
        "name": "Urban Farm",
        "description": "Urban agriculture space for food production and education",
        "icon": "🚜"
    },
    "plaza": {
        "name": "Public Plaza",
        "description": "Public gathering space with seating and amenities",
        "icon": "🏛️"
    },
    "affordable_housing": {
        "name": "Affordable Housing",
        "description": "Housing units targeted at low to moderate income households",
        "icon": "🏘️"
    },
    "mixed_income_housing": {
        "name": "Mixed-Income Housing",
        "description": "Residential development with units for various income levels",
        "icon": "🏗️"
    },
    "senior_housing": {
        "name": "Senior Housing",
        "description": "Housing designed for senior citizens with accessibility features",
        "icon": "👵"
    },
    "family_housing": {
        "name": "Family Housing",
        "description": "Multi-bedroom units suitable for families with children",
        "icon": "🏠"
    },
    "transit_oriented_housing": {
        "name": "Transit-Oriented Housing",
        "description": "Residential development near transit stops for easy commuting",
        "icon": "🚊"
    }
}


def calculate_demand_scores(tract_data: pd.Series) -> Dict[str, float]:
    """
    Calculate demand scores for different land use types based on census demographics.
    
    Args:
        tract_data: A pandas Series containing census tract demographic data
        
    Returns:
        Dictionary mapping recommendation types to demand scores (0-100)
    """
    scores = {}
    
    # Extract demographic indicators
    total_pop = tract_data.get('tract_total_pop', 0)
    median_income = tract_data.get('tract_median_income', 0)
    median_age = tract_data.get('tract_median_age', 0)
    pop_under_18 = tract_data.get('tract_pop_under_18', 0)
    pop_65_plus = tract_data.get('tract_pop_65_plus', 0)
    family_households = tract_data.get('tract_family_households', 0)
    single_person_households = tract_data.get('tract_single_person_households', 0)
    transit_commuters = tract_data.get('tract_transit_commuters', 0)
    median_rent = tract_data.get('tract_median_rent', 0)
    median_home_value = tract_data.get('tract_median_home_value', 0)
    
    # Convert to numeric, handling missing values
    total_pop = pd.to_numeric(total_pop, errors='coerce') or 0
    median_income = pd.to_numeric(median_income, errors='coerce') or 0
    median_age = pd.to_numeric(median_age, errors='coerce') or 0
    pop_under_18 = pd.to_numeric(pop_under_18, errors='coerce') or 0
    pop_65_plus = pd.to_numeric(pop_65_plus, errors='coerce') or 0
    family_households = pd.to_numeric(family_households, errors='coerce') or 0
    single_person_households = pd.to_numeric(single_person_households, errors='coerce') or 0
    transit_commuters = pd.to_numeric(transit_commuters, errors='coerce') or 0
    median_rent = pd.to_numeric(median_rent, errors='coerce') or 0
    median_home_value = pd.to_numeric(median_home_value, errors='coerce') or 0
    
    # Avoid division by zero
    if total_pop == 0:
        return {rec_type: 0.0 for rec_type in RECOMMENDATION_TYPES.keys()}
    
    # Calculate percentages
    pct_under_18 = (pop_under_18 / total_pop) * 100 if total_pop > 0 else 0
    pct_65_plus = (pop_65_plus / total_pop) * 100 if total_pop > 0 else 0
    pct_family_households = (family_households / (family_households + single_person_households)) * 100 if (family_households + single_person_households) > 0 else 0
    pct_transit_commuters = (transit_commuters / total_pop) * 100 if total_pop > 0 else 0
    
    # PARK/PLAYGROUND: High demand if many families with children
    # Score based on: high % under 18, high family households, moderate-high population density
    park_score = (
        min(pct_under_18 / 30 * 40, 40) +  # Up to 40 points for children (30%+ is high)
        min(pct_family_households / 70 * 30, 30) +  # Up to 30 points for families (70%+ is high)
        min(total_pop / 5000 * 30, 30)  # Up to 30 points for population density
    )
    scores['park'] = min(park_score, 100)
    
    # COMMUNITY GARDEN: High demand if families and moderate income
    # Score based on: family households, moderate income (not too high, not too low)
    garden_score = (
        min(pct_family_households / 70 * 35, 35) +  # Up to 35 points for families
        (min(abs(median_income - 50000) / 50000 * 30, 30) if median_income > 0 else 0) +  # Prefer moderate income
        min(total_pop / 4000 * 35, 35)  # Moderate population density
    )
    # Adjust for income preference (not too affluent, not too poor)
    if median_income > 0:
        if 30000 <= median_income <= 70000:
            garden_score += 20
        elif 20000 <= median_income <= 80000:
            garden_score += 10
    scores['community_garden'] = min(garden_score, 100)
    
    # SENIOR CENTER: High demand if many seniors
    # Score based on: high % 65+, population density
    senior_score = (
        min(pct_65_plus / 25 * 50, 50) +  # Up to 50 points for seniors (25%+ is high)
        min(total_pop / 4000 * 50, 50)  # Up to 50 points for population density
    )
    scores['senior_center'] = min(senior_score, 100)
    
    # WALKING TRAIL: High demand if seniors, families, or active population
    # Score based on: seniors, families, population density
    trail_score = (
        min(pct_65_plus / 20 * 25, 25) +  # Seniors benefit from walking
        min(pct_family_households / 60 * 25, 25) +  # Families for recreation
        min(total_pop / 3000 * 50, 50)  # Population density
    )
    scores['walking_trail'] = min(trail_score, 100)
    
    # TRANSIT-ORIENTED DEVELOPMENT: High demand if many transit commuters
    # Score based on: high % transit commuters, population density
    transit_score = (
        min(pct_transit_commuters / 40 * 60, 60) +  # Up to 60 points for transit commuters (40%+ is high)
        min(total_pop / 4000 * 40, 40)  # Population density
    )
    scores['transit_hub'] = min(transit_score, 100)
    
    # SPORTS FACILITY: High demand if families with children
    # Score based on: children, families, population density
    sports_score = (
        min(pct_under_18 / 30 * 40, 40) +  # Children need sports
        min(pct_family_households / 70 * 30, 30) +  # Family support
        min(total_pop / 5000 * 30, 30)  # Population density
    )
    scores['sports_facility'] = min(sports_score, 100)
    
    # URBAN FARM: Similar to community garden but may prefer lower income areas
    # Score based on: families, lower-moderate income, population
    urban_farm_score = (
        min(pct_family_households / 70 * 30, 30) +
        min(total_pop / 4000 * 30, 30)
    )
    # Prefer lower-moderate income areas
    if median_income > 0:
        if 20000 <= median_income <= 60000:
            urban_farm_score += 30
        elif 15000 <= median_income <= 70000:
            urban_farm_score += 20
    scores['urban_farm'] = min(urban_farm_score, 100)
    
    # PLAZA: High demand in dense areas with mixed demographics
    # Score based on: high population density, mixed demographics
    plaza_score = (
        min(total_pop / 6000 * 60, 60) +  # High population density
        min((pct_family_households + (100 - pct_family_households)) / 100 * 40, 40)  # Mixed demographics
    )
    scores['plaza'] = min(plaza_score, 100)
    
    # AFFORDABLE HOUSING: High demand in low-moderate income areas with high rent burden
    # Score based on: low-moderate income, high rent-to-income ratio, population density
    affordable_score = 0
    if median_income > 0:
        # Lower income = higher need for affordable housing
        if median_income < 40000:
            affordable_score += 40  # High need
        elif median_income < 60000:
            affordable_score += 25  # Moderate need
        elif median_income < 80000:
            affordable_score += 10  # Lower need
        
        # Rent burden: if annual rent > 30% of income, there's housing cost burden
        annual_rent = median_rent * 12 if median_rent > 0 else 0
        if annual_rent > 0 and median_income > 0:
            rent_burden_pct = (annual_rent / median_income) * 100
            if rent_burden_pct > 35:  # Severely cost-burdened
                affordable_score += 35
            elif rent_burden_pct > 30:  # Cost-burdened
                affordable_score += 25
            elif rent_burden_pct > 25:
                affordable_score += 15
    affordable_score += min(total_pop / 4000 * 25, 25)  # Population density
    scores['affordable_housing'] = min(affordable_score, 100)
    
    # MIXED-INCOME HOUSING: Good for diverse areas with moderate income levels
    # Score based on: moderate income, diverse demographics, population density
    mixed_income_score = 0
    if median_income > 0:
        # Prefer moderate income areas (not too high, not too low)
        if 40000 <= median_income <= 80000:
            mixed_income_score += 35
        elif 30000 <= median_income <= 90000:
            mixed_income_score += 25
        elif 20000 <= median_income <= 100000:
            mixed_income_score += 15
    # Mixed household types (both families and singles)
    household_diversity = min(abs(pct_family_households - 50) / 50 * 30, 30)  # More points if closer to 50/50
    mixed_income_score += (30 - household_diversity)  # Invert so balanced = higher score
    mixed_income_score += min(total_pop / 5000 * 35, 35)  # Population density
    scores['mixed_income_housing'] = min(mixed_income_score, 100)
    
    # SENIOR HOUSING: High demand in areas with many seniors
    # Score based on: high % 65+, population density, potentially lower income
    senior_housing_score = (
        min(pct_65_plus / 25 * 50, 50) +  # Up to 50 points for seniors (25%+ is high)
        min(total_pop / 3000 * 30, 30)  # Population density
    )
    # Bonus if lower income (seniors may need affordable senior housing)
    if median_income > 0 and median_income < 60000:
        senior_housing_score += 20
    scores['senior_housing'] = min(senior_housing_score, 100)
    
    # FAMILY HOUSING: High demand in areas with many families
    # Score based on: high family households, children, population density
    family_housing_score = (
        min(pct_family_households / 70 * 40, 40) +  # Up to 40 points for families
        min(pct_under_18 / 30 * 30, 30) +  # Up to 30 points for children
        min(total_pop / 4000 * 30, 30)  # Population density
    )
    scores['family_housing'] = min(family_housing_score, 100)
    
    # TRANSIT-ORIENTED HOUSING: High demand near transit
    # Score based on: transit commuters, population density, potentially younger residents
    transit_housing_score = (
        min(pct_transit_commuters / 40 * 45, 45) +  # Up to 45 points for transit commuters
        min(total_pop / 5000 * 35, 35)  # Population density
    )
    # Bonus if moderate age (working-age population more likely to use transit)
    if median_age > 0:
        if 30 <= median_age <= 50:  # Prime working age
            transit_housing_score += 20
        elif 25 <= median_age <= 55:
            transit_housing_score += 10
    scores['transit_oriented_housing'] = min(transit_housing_score, 100)
    
    return scores


def get_top_recommendations(tract_data: pd.Series, top_n: int = 3) -> List[Dict]:
    """
    Get top N recommendations for a census tract based on demand scores.
    
    Args:
        tract_data: A pandas Series containing census tract demographic data
        top_n: Number of top recommendations to return (default: 3)
        
    Returns:
        List of recommendation dictionaries sorted by score (highest first)
    """
    scores = calculate_demand_scores(tract_data)
    
    # Sort by score descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for rec_type, score in sorted_scores[:top_n]:
        if score > 0:  # Only include recommendations with positive scores
            rec_info = RECOMMENDATION_TYPES[rec_type].copy()
            rec_info['type'] = rec_type
            rec_info['demand_score'] = round(score, 2)
            rec_info['justification'] = generate_justification(rec_type, tract_data, score)
            recommendations.append(rec_info)
    
    return recommendations


def generate_justification(rec_type: str, tract_data: pd.Series, score: float) -> str:
    """
    Generate a human-readable justification for a recommendation.
    
    Args:
        rec_type: Type of recommendation
        tract_data: Census tract demographic data
        score: Demand score for this recommendation
        
    Returns:
        String explaining why this recommendation was made
    """
    total_pop = pd.to_numeric(tract_data.get('tract_total_pop', 0), errors='coerce') or 0
    pop_under_18 = pd.to_numeric(tract_data.get('tract_pop_under_18', 0), errors='coerce') or 0
    pop_65_plus = pd.to_numeric(tract_data.get('tract_pop_65_plus', 0), errors='coerce') or 0
    family_households = pd.to_numeric(tract_data.get('tract_family_households', 0), errors='coerce') or 0
    single_person_households = pd.to_numeric(tract_data.get('tract_single_person_households', 0), errors='coerce') or 0
    transit_commuters = pd.to_numeric(tract_data.get('tract_transit_commuters', 0), errors='coerce') or 0
    median_income = pd.to_numeric(tract_data.get('tract_median_income', 0), errors='coerce') or 0
    median_rent = pd.to_numeric(tract_data.get('tract_median_rent', 0), errors='coerce') or 0
    median_age = pd.to_numeric(tract_data.get('tract_median_age', 0), errors='coerce') or 0
    
    pct_under_18 = (pop_under_18 / total_pop * 100) if total_pop > 0 else 0
    pct_65_plus = (pop_65_plus / total_pop * 100) if total_pop > 0 else 0
    
    # Calculate percentage of family households
    total_households = family_households + single_person_households
    pct_family_households = (family_households / total_households * 100) if total_households > 0 else 0
    
    justifications = []
    
    if rec_type == 'park':
        if pct_under_18 > 20:
            justifications.append(f"{pct_under_18:.1f}% of the population is under 18 years old")
        if family_households > 0:
            justifications.append(f"{family_households:.0f} family households would benefit")
        if total_pop > 3000:
            justifications.append(f"High population density ({total_pop:.0f} residents)")
            
    elif rec_type == 'community_garden':
        if family_households > 500:
            justifications.append(f"Many family households ({family_households:.0f})")
        if 30000 <= median_income <= 70000:
            justifications.append(f"Moderate income level supports community engagement")
        if total_pop > 2000:
            justifications.append(f"Sufficient population ({total_pop:.0f} residents) for community participation")
            
    elif rec_type == 'senior_center':
        if pct_65_plus > 15:
            justifications.append(f"{pct_65_plus:.1f}% of the population is 65+ years old")
        if pop_65_plus > 500:
            justifications.append(f"{pop_65_plus:.0f} senior residents would benefit")
        if total_pop > 2000:
            justifications.append(f"Sufficient population base ({total_pop:.0f} residents)")
            
    elif rec_type == 'walking_trail':
        if pct_65_plus > 15:
            justifications.append(f"{pct_65_plus:.1f}% seniors who benefit from walking")
        if pct_under_18 > 15:
            justifications.append(f"{pct_under_18:.1f}% children for recreation")
        if total_pop > 2000:
            justifications.append(f"Dense population ({total_pop:.0f} residents) would use the trail")
            
    elif rec_type == 'transit_hub':
        if transit_commuters > 200:
            justifications.append(f"{transit_commuters:.0f} transit commuters in the area")
        if total_pop > 3000:
            justifications.append(f"High population density ({total_pop:.0f} residents)")
            
    elif rec_type == 'sports_facility':
        if pct_under_18 > 20:
            justifications.append(f"{pct_under_18:.1f}% of the population is under 18")
        if family_households > 500:
            justifications.append(f"{family_households:.0f} family households would benefit")
            
    elif rec_type == 'urban_farm':
        if family_households > 500:
            justifications.append(f"Many family households ({family_households:.0f})")
        if 20000 <= median_income <= 60000:
            justifications.append(f"Moderate income level suitable for urban agriculture")
            
    elif rec_type == 'plaza':
        if total_pop > 4000:
            justifications.append(f"High population density ({total_pop:.0f} residents)")
        justifications.append("Diverse community would benefit from public gathering space")
    
    elif rec_type == 'affordable_housing':
        if median_income > 0 and median_income < 60000:
            justifications.append(f"Median household income of ${median_income:,.0f} indicates need for affordable options")
        if median_rent > 0 and median_income > 0:
            annual_rent = median_rent * 12
            rent_burden = (annual_rent / median_income) * 100 if median_income > 0 else 0
            if rent_burden > 30:
                justifications.append(f"High rent burden ({rent_burden:.1f}% of income spent on rent)")
        if total_pop > 2000:
            justifications.append(f"Sufficient population ({total_pop:.0f} residents) to support new housing")
            
    elif rec_type == 'mixed_income_housing':
        if 40000 <= median_income <= 80000:
            justifications.append(f"Moderate income area (${median_income:,.0f}) suitable for diverse housing")
        if total_pop > 3000:
            justifications.append(f"Dense population ({total_pop:.0f} residents) supports mixed-income development")
        justifications.append("Diverse community composition benefits from housing variety")
            
    elif rec_type == 'senior_housing':
        if pct_65_plus > 15:
            justifications.append(f"{pct_65_plus:.1f}% of the population is 65+ years old")
        if pop_65_plus > 500:
            justifications.append(f"{pop_65_plus:.0f} senior residents would benefit from age-appropriate housing")
        if median_income > 0 and median_income < 60000:
            justifications.append("Lower income suggests need for affordable senior housing options")
            
    elif rec_type == 'family_housing':
        if pct_family_households > 50:
            justifications.append(f"{pct_family_households:.1f}% of households are families")
        if family_households > 500:
            justifications.append(f"{family_households:.0f} family households would benefit from larger units")
        if pct_under_18 > 20:
            justifications.append(f"{pct_under_18:.1f}% children indicating strong family presence")
        if total_pop > 3000:
            justifications.append(f"High population density ({total_pop:.0f} residents) supports family housing demand")
            
    elif rec_type == 'transit_oriented_housing':
        if transit_commuters > 200:
            justifications.append(f"{transit_commuters:.0f} transit commuters would benefit from nearby housing")
        if total_pop > 3000:
            justifications.append(f"High population density ({total_pop:.0f} residents) near transit")
        if median_age > 0 and 30 <= median_age <= 50:
            justifications.append(f"Working-age population (median age {median_age:.0f}) likely to use transit")
    
    if not justifications:
        return f"Based on demographic analysis, this area shows moderate demand (score: {score:.1f}/100)."
    
    return " • ".join(justifications) + f" (Demand score: {score:.1f}/100)"


def analyze_parcel_recommendations(
    enriched_data_path: str,
    output_path: Optional[str] = None,
    min_score: float = 20.0,
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Analyze all parcels in enriched data and generate recommendations.
    
    Args:
        enriched_data_path: Path to the enriched parcels CSV file
        output_path: Optional path to save results (default: add _recommendations suffix)
        min_score: Minimum demand score to include a recommendation (default: 20.0)
        sample_size: Optional limit on number of parcels to process (for testing)
        
    Returns:
        DataFrame with parcels and their top recommendations
    """
    print(f"Loading enriched parcel data from {enriched_data_path}...")
    
    if sample_size:
        df = pd.read_csv(enriched_data_path, nrows=sample_size)
        print(f"⚠️  SAMPLE MODE: Processing only {sample_size:,} parcels")
    else:
        df = pd.read_csv(enriched_data_path)
    
    print(f"Loaded {len(df):,} parcels")
    
    # Filter to parcels with census data
    census_cols = ['tract_total_pop', 'tract_median_income', 'tract_pop_under_18']
    has_census = df[census_cols].notna().any(axis=1)
    df_with_census = df[has_census].copy()
    
    print(f"Parcels with census data: {len(df_with_census):,} ({len(df_with_census)/len(df)*100:.1f}%)")
    
    if len(df_with_census) == 0:
        print("ERROR: No parcels with census data found!")
        return pd.DataFrame()
    
    # Generate recommendations for each unique census tract
    print("\nGenerating recommendations for each census tract...")
    
    # Group by census tract to avoid redundant calculations
    tract_recommendations = {}
    
    unique_tracts = df_with_census['census_tract'].dropna().unique()
    print(f"Processing {len(unique_tracts):,} unique census tracts...")
    
    for i, tract in enumerate(unique_tracts):
        if i % 100 == 0 and i > 0:
            print(f"  Processed {i:,}/{len(unique_tracts):,} tracts...")
        
        tract_data = df_with_census[df_with_census['census_tract'] == tract].iloc[0]
        recommendations = get_top_recommendations(tract_data, top_n=3)
        
        # Filter by minimum score
        recommendations = [r for r in recommendations if r['demand_score'] >= min_score]
        
        if recommendations:
            tract_recommendations[tract] = recommendations
    
    print(f"Generated recommendations for {len(tract_recommendations):,} tracts")
    
    # Expand recommendations to parcel level
    print("\nAssigning recommendations to parcels...")
    results = []
    
    for idx, row in df_with_census.iterrows():
        tract = row.get('census_tract')
        
        if pd.isna(tract) or tract not in tract_recommendations:
            continue
        
        recommendations = tract_recommendations[tract]
        
        for rec in recommendations:
            result_row = {
                'parcel_number': row.get('parcel_number'),
                'location': row.get('location'),
                'census_tract': tract,
                'recommendation_type': rec['type'],
                'recommendation_name': rec['name'],
                'recommendation_description': rec['description'],
                'demand_score': rec['demand_score'],
                'justification': rec['justification'],
                'icon': rec.get('icon', ''),
                # Include key demographics for reference
                'tract_total_pop': row.get('tract_total_pop'),
                'tract_pop_under_18': row.get('tract_pop_under_18'),
                'tract_pop_65_plus': row.get('tract_pop_65_plus'),
                'tract_median_income': row.get('tract_median_income'),
                'tract_family_households': row.get('tract_family_households'),
            }
            results.append(result_row)
    
    results_df = pd.DataFrame(results)
    print(f"Generated {len(results_df):,} parcel-recommendation pairs")
    
    # Save results if output path specified
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved recommendations to {output_path}")
    elif not sample_size:
        # Auto-generate output path
        base_path = enriched_data_path.replace('.csv', '')
        output_path = f"{base_path}_recommendations.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved recommendations to {output_path}")
    
    return results_df


def get_recommendations_by_category(recommendations: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Categorize recommendations into housing and public space types.
    
    Args:
        recommendations: List of recommendation dictionaries
        
    Returns:
        Dictionary with 'housing' and 'public_spaces' keys containing filtered lists
    """
    housing = [r for r in recommendations if r['type'] in HOUSING_TYPES]
    public_spaces = [r for r in recommendations if r['type'] in PUBLIC_SPACE_TYPES]
    
    return {
        'housing': housing,
        'public_spaces': public_spaces
    }


def get_recommendations_for_parcel(
    parcel_number: str,
    enriched_data_path: str
) -> Optional[List[Dict]]:
    """
    Get recommendations for a specific parcel by parcel number.
    
    Args:
        parcel_number: The parcel number to look up
        enriched_data_path: Path to the enriched parcels CSV file
        
    Returns:
        List of recommendation dictionaries, or None if parcel not found
    """
    df = pd.read_csv(enriched_data_path)
    
    # Find the parcel
    parcel_df = df[df['parcel_number'].astype(str) == str(parcel_number)]
    
    if len(parcel_df) == 0:
        return None
    
    parcel_data = parcel_df.iloc[0]
    
    # Check if census data is available
    if pd.isna(parcel_data.get('census_tract')):
        return []
    
    # Get recommendations
    recommendations = get_top_recommendations(parcel_data, top_n=3)
    
    return recommendations


if __name__ == "__main__":
    # Example usage
    import os
    
    # Get the path to the enriched data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    enriched_data_path = os.path.normpath(
        os.path.join(current_dir, "../../data/philadelphia_parcels_enriched.csv")
    )
    
    if os.path.exists(enriched_data_path):
        print("=" * 70)
        print("DEMAND MODEL - LAND USE RECOMMENDATIONS")
        print("=" * 70)
        print()
        
        # Run analysis with sample size for testing
        results = analyze_parcel_recommendations(
            enriched_data_path,
            sample_size=1000,  # Process 1000 parcels for testing
            min_score=25.0  # Only show recommendations with score >= 25
        )
        
        if len(results) > 0:
            print("\n" + "=" * 70)
            print("SAMPLE RECOMMENDATIONS")
            print("=" * 70)
            
            # Show summary by recommendation type
            print("\nRecommendations by type:")
            print(results.groupby('recommendation_type')['parcel_number'].count().sort_values(ascending=False))
            
            # Show top recommendations
            print("\nTop 10 recommendations (by demand score):")
            top_recs = results.nlargest(10, 'demand_score')[
                ['parcel_number', 'location', 'recommendation_name', 'demand_score', 'justification']
            ]
            print(top_recs.to_string(index=False))
            
            # Show example recommendations for a specific parcel
            if len(results) > 0:
                sample_parcel = results.iloc[0]['parcel_number']
                print(f"\n{'=' * 70}")
                print(f"Example: Recommendations for parcel {sample_parcel}")
                print("=" * 70)
                parcel_recs = results[results['parcel_number'] == sample_parcel][
                    ['recommendation_name', 'demand_score', 'justification']
                ]
                print(parcel_recs.to_string(index=False))
    else:
        print(f"ERROR: Enriched data file not found at {enriched_data_path}")
        print("Please run the ETL pipeline first to generate enriched data.")

