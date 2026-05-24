import pandas as pd
import os


def get_harga_sayuran(nama_sayur: str, daerah: str) -> str:
    """Mendapatkan harga sayuran berdasarkan nama sayuran dan nama daerah/kota.

    Args:
        nama_sayur: Nama sayuran yang ingin dicek harganya (contoh: Cabai Merah, Tomat, Bawang Merah)
        daerah: Nama daerah atau kota (contoh: Jakarta, Bandung, Surabaya, Yogyakarta, Medan, Makassar)

    Returns:
        String berisi informasi harga sayuran di daerah tersebut.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "data", "harga_sayuran.csv")
    df = pd.read_csv(csv_path)

    # Exact match (case-insensitive)
    mask = (
        df["nama_sayur"].str.lower() == nama_sayur.lower()
    ) & (
        df["daerah"].str.lower() == daerah.lower()
    )
    result = df[mask]

    # Fallback: partial match
    if result.empty:
        mask = (
            df["nama_sayur"].str.lower().str.contains(nama_sayur.lower(), na=False)
        ) & (
            df["daerah"].str.lower().str.contains(daerah.lower(), na=False)
        )
        result = df[mask]

    if result.empty:
        # Coba cari sayuran yang tersedia
        sayuran_tersedia = df["nama_sayur"].unique().tolist()
        daerah_tersedia = df["daerah"].unique().tolist()
        return (
            f"Maaf, data harga untuk '{nama_sayur}' di '{daerah}' tidak tersedia. "
            f"Sayuran yang tersedia: {', '.join(sayuran_tersedia)}. "
            f"Daerah yang tersedia: {', '.join(daerah_tersedia)}."
        )

    row = result.iloc[0]
    harga_formatted = f"Rp {int(row['harga_per_kg']):,}".replace(",", ".")
    return (
        f"Harga {row['nama_sayur']} di {row['daerah']}: "
        f"{harga_formatted}/{row['satuan']} "
        f"(terakhir diperbarui: {row['tanggal_update']})"
    )
