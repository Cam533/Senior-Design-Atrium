DROP POLICY IF EXISTS "Allow authenticated uploads to avatars" ON storage.objects;
DROP POLICY IF EXISTS "Allow authenticated update own avatar" ON storage.objects;
DROP POLICY IF EXISTS "Allow public read avatars" ON storage.objects;

-- 1. INSERT: allow uploads to the avatars bucket (this fixes "new row violates RLS")
CREATE POLICY "Allow authenticated uploads to avatars"
  ON storage.objects
  FOR INSERT
  TO authenticated
  WITH CHECK (bucket_id = 'avatars');

-- 2. UPDATE: allow overwriting a file (so changing profile pic works)
CREATE POLICY "Allow authenticated update own avatar"
  ON storage.objects
  FOR UPDATE
  TO authenticated
  USING (bucket_id = 'avatars')
  WITH CHECK (bucket_id = 'avatars');

-- 3. SELECT: allow reading object info (needed for listing/getting URL)
CREATE POLICY "Allow public read avatars"
  ON storage.objects
  FOR SELECT
  TO authenticated
  USING (bucket_id = 'avatars');
