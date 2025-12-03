# Atrium

**Atrium** is an AI-powered decision-support platform that helps cities, planners, developers, and residents better understand and activate vacant or underused urban lots.

Using geospatial data, zoning regulations, census information, and large language models, Atrium surfaces actionable insights about what *can* be built, what *should* be built, and what is most feasible given local demand, sustainability goals, and equity considerations.

**Demo Video:** *(link coming soon)*

---

## Problem

Cities contain many vacant or underutilized parcels that could support housing, green space, or community infrastructure. However, understanding development feasibility requires navigating fragmented data sources, complex zoning codes, and limited planning expertise.

Atrium addresses this by:

* Identifying vacant and underused lots using parcel-level data
* Providing zoning- and code-aware insights through an AI chat interface
* Recommending potential reuses that balance demand, feasibility, sustainability, and equity
* Laying the groundwork to connect realty groups, architects, and contractors based on relevant experience

---

## What We’ve Built So Far

### Core User Experience

* Interactive frontend with map-based parcel exploration
* AI-powered parcel-level chat grounded in Philadelphia zoning and building codes

### Data Infrastructure

* Deployed an AWS relational database for census and demographic data
* Populated the map with parcel boundaries and vacant lot indicators

### Modeling & Analytics

* Developed three parcel-level environmental and location scores (1–10), including walkability
* Initiated a demand prediction model using census and location-based features

---

## Planned Features & Next Steps

### User Accounts & Personalization

* User authentication and profile management
* Ability to save parcels, insights, and recommendations

### Collaboration & Social Features

* Lightweight social tools to share and discuss development ideas

### LLM & RAG Enhancements

* Expanded datasets for retrieval-augmented generation
* Improved contextual grounding for zoning, policy, and environmental constraints
* Increased robustness and reliability of the chat experience

### Data Visualization

* Display census and demographic data directly in the frontend at the parcel level

