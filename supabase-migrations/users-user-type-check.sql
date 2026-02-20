-- Fix users_user_type_check so user_type accepts the app's values
-- Run in Supabase Dashboard → SQL Editor → New query → paste & Run

ALTER TABLE public.users
  DROP CONSTRAINT IF EXISTS users_user_type_check;

ALTER TABLE public.users
  ADD CONSTRAINT users_user_type_check CHECK (
    user_type IN ('resident', 'business', 'city_planner', 'nonprofit', 'other')
  );
