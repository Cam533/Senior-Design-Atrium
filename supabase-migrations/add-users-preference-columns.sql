-- Add email preference columns to public.users
-- Run in Supabase Dashboard → SQL Editor → New query → paste & Run

ALTER TABLE public.users
  ADD COLUMN IF NOT EXISTS email_plot_updates boolean DEFAULT true,
  ADD COLUMN IF NOT EXISTS email_product_news boolean DEFAULT false,
  ADD COLUMN IF NOT EXISTS unsubscribe_all boolean DEFAULT false;
