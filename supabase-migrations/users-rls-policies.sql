-- Allow authenticated users to manage their own row in public.users
-- Run in Supabase Dashboard → SQL Editor → New query → paste & Run

ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can insert own row" ON public.users;
CREATE POLICY "Users can insert own row"
  ON public.users FOR INSERT TO authenticated
  WITH CHECK (auth.uid() = id);

DROP POLICY IF EXISTS "Users can update own row" ON public.users;
CREATE POLICY "Users can update own row"
  ON public.users FOR UPDATE TO authenticated
  USING (auth.uid() = id) WITH CHECK (auth.uid() = id);

DROP POLICY IF EXISTS "Users can select own row" ON public.users;
CREATE POLICY "Users can select own row"
  ON public.users FOR SELECT TO authenticated
  USING (auth.uid() = id);
