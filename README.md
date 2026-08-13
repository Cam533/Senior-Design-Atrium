# Atrium

**Atrium** is an AI-powered decision-support platform that helps cities, planners, developers, and residents better understand and activate vacant or underused urban lots. For now, we are focusing on the city of Philadelphia, PA.

Using geospatial data, zoning regulations, census information, and large language models, Atrium surfaces actionable insights about what *can* be built, what *should* be built, and what is most feasible given local demand, sustainability goals, and equity considerations.

---

## Problem

Cities contain many vacant or underutilized plots that could support housing, green space, or community infrastructure. However, understanding development feasibility requires navigating fragmented data sources, complex zoning codes, and limited planning expertise.

Atrium addresses this by:

* Identifying vacant and underused lots using city-level building and land data
* Providing zoning- and code-aware insights through an AI chat interface
* Recommending potential reuses that balance demand, feasibility, sustainability, and equity
* Laying the groundwork to connect realty groups, architects, and contractors based on relevant experience

---

## What We’ve Built So Far

### Core User Experience

* Interactive frontend with map-based lot exploration
* AI-powered lot-level chat grounded in Philadelphia zoning and building codes

### Data Infrastructure

* Deployed an AWS relational database for census and demographic data
* Populated the map with plot boundaries and vacant lot indicators

### Modeling & Analytics

* Developed lot-level environmental and location scores (1–10), including walkability, transit, and recreational
* Initiated a demand prediction model using census and location-based features

#### Homepage
<img width="1439" height="774" alt="Atrium Homepage" src="https://github.com/user-attachments/assets/3e0297ab-7d10-4883-ab00-ab1c05b4ab87" />

---

## Planned Features & Next Steps

### User Accounts & Personalization

* User authentication and profile management
* Ability to save addresses, insights, and recommendations

**Delete account (Supabase):** The backend deletes the user in Supabase Auth and cleans up their row in `public.users` and their avatar in Storage. To enable it:

1. In [Supabase Dashboard](https://supabase.com/dashboard) → your project → **Settings** → **API**: copy **Project URL** and **service_role** key (under "Project API keys").
2. Set environment variables for the backend:
   - `SUPABASE_URL` = Project URL (e.g. `https://xxxx.supabase.co`)
   - `SUPABASE_SERVICE_ROLE_KEY` = service_role key (keep secret; never expose in the frontend)
3. Restart the backend. The "Permanently delete my account" button on the Profile → Security tab will then work.

**Profile pictures (avatars bucket):** Uploads use Supabase Storage. Create the bucket once:

1. In [Supabase Dashboard](https://supabase.com/dashboard) → your project → **Storage**.
2. Click **New bucket**. Name it exactly: `avatars`.
3. Turn **Public bucket** ON (so profile image URLs work).
4. Click **Create bucket**.
5. Open the `avatars` bucket → **Policies** → **New policy**. Use "For full customization" and add:
   - **Policy name:** Allow authenticated uploads
   - **Allowed operation:** INSERT (and SELECT if you want read via API; public bucket already allows public read).
   - **Target roles:** authenticated (or leave default).
   - **Policy definition:** `true` for authenticated users, or use: `bucket_id = 'avatars' AND auth.role() = 'authenticated'`.
   Or use the template "Allow authenticated uploads" if available and scope it to bucket `avatars`.

After the bucket exists and is public, profile photo upload on signup and Profile page will work.

### Collaboration & Social Features

* Lightweight social tools to share and discuss development ideas

### LLM & RAG Enhancements

* Expanded datasets for retrieval-augmented generation
* Improved contextual grounding for zoning, policy, and environmental constraints
* Increased robustness and reliability of the chat experience

### Data Visualization

* Display census and demographic data directly in the frontend at greater depth

