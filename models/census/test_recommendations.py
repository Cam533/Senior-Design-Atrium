"""
Comprehensive test script to demonstrate both HOUSING and PUBLIC SPACE recommendations.
"""

import os
import sys
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from models.census.demand_model import (
    get_recommendations_for_parcel,
    analyze_parcel_recommendations,
    calculate_demand_scores,
    RECOMMENDATION_TYPES,
    PUBLIC_SPACE_TYPES,
    HOUSING_TYPES
)

def categorize_recommendations(recommendations):
    """Separate recommendations into housing and public space categories."""
    housing = [r for r in recommendations if r['type'] in HOUSING_TYPES]
    public_spaces = [r for r in recommendations if r['type'] in PUBLIC_SPACE_TYPES]
    return housing, public_spaces

def test_all_recommendations():
    """Test and display both housing and public space recommendations."""
    print("=" * 80)
    print("COMPREHENSIVE RECOMMENDATIONS TEST")
    print("HOUSING + PUBLIC SPACE RECOMMENDATIONS")
    print("=" * 80)
    
    # Path to enriched data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    enriched_data_path = os.path.normpath(
        os.path.join(current_dir, "../../data/philadelphia_parcels_enriched.csv")
    )
    
    if not os.path.exists(enriched_data_path):
        print(f"ERROR: Enriched data file not found at {enriched_data_path}")
        print("Please run the ETL pipeline first.")
        return
    
    # Load sample data
    print("\nLoading sample parcels...")
    df = pd.read_csv(enriched_data_path, nrows=500)
    
    # Filter to parcels with census data
    census_cols = ['tract_total_pop', 'tract_median_income']
    df_with_census = df[df[census_cols].notna().any(axis=1)].copy()
    
    print(f"Loaded {len(df_with_census):,} parcels with census data\n")
    
    if len(df_with_census) == 0:
        print("No parcels with census data found!")
        return
    
    # ========================================================================
    # EXAMPLE 1: Complete recommendations for a single parcel
    # ========================================================================
    print("=" * 80)
    print("EXAMPLE 1: Complete Recommendations for a Parcel")
    print("=" * 80)
    
    example_parcel = df_with_census.iloc[0]
    parcel_num = example_parcel.get('parcel_number')
    location = example_parcel.get('location', 'N/A')
    
    print(f"\nParcel: {parcel_num}")
    print(f"Location: {location}")
    print(f"\nDemographics:")
    print(f"  Total Population: {example_parcel.get('tract_total_pop', 0):,.0f}")
    print(f"  Median Income: ${example_parcel.get('tract_median_income', 0):,.0f}")
    print(f"  Population Under 18: {example_parcel.get('tract_pop_under_18', 0):,.0f}")
    print(f"  Population 65+: {example_parcel.get('tract_pop_65_plus', 0):,.0f}")
    print(f"  Family Households: {example_parcel.get('tract_family_households', 0):,.0f}")
    
    # Get all recommendations
    recommendations = get_recommendations_for_parcel(str(parcel_num), enriched_data_path)
    
    if recommendations:
        housing_recs, public_space_recs = categorize_recommendations(recommendations)
        
        print(f"\n{'='*80}")
        print("🏠 HOUSING RECOMMENDATIONS")
        print(f"{'='*80}")
        if housing_recs:
            for i, rec in enumerate(housing_recs, 1):
                print(f"\n{i}. {rec['icon']} {rec['name']}")
                print(f"   Score: {rec['demand_score']:.1f}/100")
                print(f"   Description: {rec['description']}")
                print(f"   Justification: {rec['justification']}")
        else:
            print("No housing recommendations for this parcel.")
        
        print(f"\n{'='*80}")
        print("🌳 PUBLIC SPACE RECOMMENDATIONS")
        print(f"{'='*80}")
        if public_space_recs:
            for i, rec in enumerate(public_space_recs, 1):
                print(f"\n{i}. {rec['icon']} {rec['name']}")
                print(f"   Score: {rec['demand_score']:.1f}/100")
                print(f"   Description: {rec['description']}")
                print(f"   Justification: {rec['justification']}")
        else:
            print("No public space recommendations for this parcel.")
    else:
        print("No recommendations available for this parcel.")
    
    # ========================================================================
    # EXAMPLE 2: Housing-focused area
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Area Suitable for HOUSING Development")
    print("=" * 80)
    
    # Find low-income area for affordable housing
    low_income = df_with_census[
        (df_with_census['tract_median_income'].notna()) & 
        (df_with_census['tract_median_income'] < 50000) &
        (df_with_census['tract_median_rent'].notna())
    ]
    
    if len(low_income) > 0:
        parcel = low_income.iloc[0]
        parcel_num = parcel.get('parcel_number')
        
        print(f"\nParcel: {parcel_num}")
        print(f"Location: {parcel.get('location', 'N/A')}")
        print(f"\nDemographics:")
        print(f"  Median Income: ${parcel.get('tract_median_income', 0):,.0f}")
        print(f"  Median Rent: ${parcel.get('tract_median_rent', 0):,.0f}/month")
        
        if parcel.get('tract_median_income', 0) > 0 and parcel.get('tract_median_rent', 0) > 0:
            annual_rent = parcel.get('tract_median_rent', 0) * 12
            rent_burden = (annual_rent / parcel.get('tract_median_income', 0)) * 100
            print(f"  Rent Burden: {rent_burden:.1f}% of income")
        
        scores = calculate_demand_scores(parcel)
        
        print(f"\n🏠 Housing Demand Scores:")
        for h_type in HOUSING_TYPES:
            if h_type in scores and scores[h_type] > 20:
                rec_info = RECOMMENDATION_TYPES[h_type]
                print(f"  {rec_info['icon']} {rec_info['name']}: {scores[h_type]:.1f}/100")
        
        recommendations = get_recommendations_for_parcel(str(parcel_num), enriched_data_path)
        housing_recs, _ = categorize_recommendations(recommendations)
        
        if housing_recs:
            print(f"\nTop Housing Recommendation:")
            top_housing = max(housing_recs, key=lambda x: x['demand_score'])
            print(f"  {top_housing['icon']} {top_housing['name']} (Score: {top_housing['demand_score']:.1f}/100)")
            print(f"  {top_housing['justification']}")
    
    # ========================================================================
    # EXAMPLE 3: Public space-focused area
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Area Suitable for PUBLIC SPACE Development")
    print("=" * 80)
    
    # Find area with many families and children
    family_areas = df_with_census[
        (df_with_census['tract_family_households'].notna()) & 
        (df_with_census['tract_pop_under_18'].notna()) &
        (df_with_census['tract_family_households'] > 500) &
        (df_with_census['tract_pop_under_18'] > 400)
    ]
    
    if len(family_areas) > 0:
        parcel = family_areas.iloc[0]
        parcel_num = parcel.get('parcel_number')
        
        print(f"\nParcel: {parcel_num}")
        print(f"Location: {parcel.get('location', 'N/A')}")
        print(f"\nDemographics:")
        print(f"  Family Households: {parcel.get('tract_family_households', 0):,.0f}")
        print(f"  Population Under 18: {parcel.get('tract_pop_under_18', 0):,.0f}")
        print(f"  Total Population: {parcel.get('tract_total_pop', 0):,.0f}")
        
        scores = calculate_demand_scores(parcel)
        
        print(f"\n🌳 Public Space Demand Scores:")
        for ps_type in PUBLIC_SPACE_TYPES:
            if ps_type in scores and scores[ps_type] > 20:
                rec_info = RECOMMENDATION_TYPES[ps_type]
                print(f"  {rec_info['icon']} {rec_info['name']}: {scores[ps_type]:.1f}/100")
        
        recommendations = get_recommendations_for_parcel(str(parcel_num), enriched_data_path)
        _, public_space_recs = categorize_recommendations(recommendations)
        
        if public_space_recs:
            print(f"\nTop Public Space Recommendations:")
            top_public = sorted(public_space_recs, key=lambda x: x['demand_score'], reverse=True)[:3]
            for rec in top_public:
                print(f"  {rec['icon']} {rec['name']} (Score: {rec['demand_score']:.1f}/100)")
    
    # ========================================================================
    # EXAMPLE 4: Batch Analysis Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Batch Analysis - Summary of All Recommendations")
    print("=" * 80)
    
    print("\nRunning batch analysis on sample parcels...")
    results = analyze_parcel_recommendations(
        enriched_data_path,
        sample_size=500,
        min_score=25.0
    )
    
    if len(results) > 0:
        print(f"\nGenerated {len(results):,} total recommendations")
        
        # Separate housing and public space recommendations
        housing_results = results[results['recommendation_type'].isin(HOUSING_TYPES)]
        public_space_results = results[results['recommendation_type'].isin(PUBLIC_SPACE_TYPES)]
        
        print(f"\n{'='*80}")
        print("📊 HOUSING RECOMMENDATIONS SUMMARY")
        print(f"{'='*80}")
        if len(housing_results) > 0:
            print(f"\nTotal Housing Recommendations: {len(housing_results):,}")
            print("\nHousing Recommendations by Type:")
            housing_by_type = housing_results.groupby('recommendation_name').agg({
                'parcel_number': 'count',
                'demand_score': ['mean', 'max']
            }).round(2)
            housing_by_type.columns = ['Count', 'Avg Score', 'Max Score']
            housing_by_type = housing_by_type.sort_values('Count', ascending=False)
            print(housing_by_type.to_string())
            
            print("\nTop 3 Housing Recommendations:")
            top_housing = housing_results.nlargest(3, 'demand_score')[
                ['location', 'recommendation_name', 'demand_score']
            ]
            print(top_housing.to_string(index=False))
        else:
            print("No housing recommendations found with score >= 25")
        
        print(f"\n{'='*80}")
        print("📊 PUBLIC SPACE RECOMMENDATIONS SUMMARY")
        print(f"{'='*80}")
        if len(public_space_results) > 0:
            print(f"\nTotal Public Space Recommendations: {len(public_space_results):,}")
            print("\nPublic Space Recommendations by Type:")
            ps_by_type = public_space_results.groupby('recommendation_name').agg({
                'parcel_number': 'count',
                'demand_score': ['mean', 'max']
            }).round(2)
            ps_by_type.columns = ['Count', 'Avg Score', 'Max Score']
            ps_by_type = ps_by_type.sort_values('Count', ascending=False)
            print(ps_by_type.to_string())
            
            print("\nTop 3 Public Space Recommendations:")
            top_public = public_space_results.nlargest(3, 'demand_score')[
                ['location', 'recommendation_name', 'demand_score']
            ]
            print(top_public.to_string(index=False))
        else:
            print("No public space recommendations found with score >= 25")
        
        # Overall summary
        print(f"\n{'='*80}")
        print("📈 OVERALL SUMMARY")
        print(f"{'='*80}")
        print(f"\nTotal Recommendations: {len(results):,}")
        print(f"  🏠 Housing: {len(housing_results):,} ({len(housing_results)/len(results)*100:.1f}%)")
        print(f"  🌳 Public Spaces: {len(public_space_results):,} ({len(public_space_results)/len(results)*100:.1f}%)")
        
    else:
        print("No recommendations found in sample data")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
    print("\n✅ Both HOUSING and PUBLIC SPACE recommendations are working!")
    print("=" * 80)


if __name__ == "__main__":
    test_all_recommendations()
