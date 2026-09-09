import numpy as np

# Sortu kalibratzerako 100 audio espektrograma adibide (1, 16, 96) neurrikoak
# Zure audioen balioak 0 eta 1 artean badoaz, np.random.rand nahiko da
calib_data = np.random.rand(100, 1, 16, 96).astype(np.float32)

# Gorde datuak numpy fitxategi batean
np.save("embeddings_calib.npy", calib_data)
print("Kalibratze fitxategia ongi gorde da: embeddings_calib.npy")
