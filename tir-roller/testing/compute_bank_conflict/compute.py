for tx in range(32):
    store = ((((tx >> 2) * 32)) + (((tx & 3) ^ (tx >> 3)) * 8))
    bank = store // 2
    bank = bank % 32
    print(tx, bank)