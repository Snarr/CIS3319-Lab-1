from crypto import DES, KeyManager

key_manager = KeyManager()

# key = key_manager.generate_key(64)

# key_manager.save_key("key.txt", key)

key = key_manager.read_key("key.txt")

des = DES(key)

subkey = des.generate_subkey(key)
print(subkey)