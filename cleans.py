from datetime import datetime, timedelta, timezone

def purge_expired_guest_files(supabase_client):
    """Finds and deletes any file in guest folders that is older than 24 hours."""
    try:
        # 1. List the root folders/items inside your bucket
        bucket_items = supabase_client.storage.from_("PDFs").list()
        
        for item in bucket_items:
            folder_name = item.get("name", "")
            
            # Target only folders prefixed with 'guest_'
            if folder_name.startswith("guest_"):
                
                # 2. Inspect the files inside this specific guest folder
                guest_files = supabase_client.storage.from_("PDFs").list(path=folder_name)
                
                for file in guest_files:
                    created_at_str = file.get("created_at")  # ISO timestamp from Supabase
                    
                    if created_at_str:
                        # Convert Supabase ISO timestamp string to a timezone-aware Python datetime object
                        # (Replacing 'Z' with UTC offset syntax for seamless parsing)
                        file_time = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                        current_time = datetime.now(timezone.utc)
                        
                        if current_time - file_time > timedelta(hours=48):
                            full_storage_path = f"{folder_name}/{file['name']}"
                            
                            # Vaporize it from Supabase
                            supabase_client.storage.from_("PDFs").remove([full_storage_path])
                            print(f"Deleted expired file: {full_storage_path}")
            else:
                print("No folders to clean up.")
                            
    except Exception as e:
        # Fails silently in production so it never crashes your user interface
        print(f"Background cleanup exception: {e}")