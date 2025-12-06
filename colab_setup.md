# 🚀 VS Code'dan Google Colab'a Bağlanma Rehberi

## Yöntem 1: Colab Eklentisi ile Doğrudan Bağlantı

### Adım 1: Colab'da Session Başlat
1. Tarayıcıda Colab'ı açın: https://colab.research.google.com
2. Yeni bir notebook oluşturun veya mevcut birini açın
3. Aşağıdaki kodu çalıştırın:

```python
# Colab'ı VS Code'a bağlamak için
from google.colab import auth
auth.authenticate_user()

# Jupyter server bilgilerini al
!jupyter notebook list
```

### Adım 2: VS Code'da Bağlantı Kur
1. VS Code'da `colab_runner.ipynb` dosyasını açın
2. Sağ üstte **"Select Kernel"** tıklayın
3. **"Existing Jupyter Server"** seçin
4. Colab'dan aldığınız URL'yi girin

---

## Yöntem 2: Ngrok ile Tunnel (Daha Kolay)

### Adım 1: Colab'da Ngrok Setup
Colab'da şu kodu çalıştırın:

```python
# Colab'da çalıştırın
!pip install -q jupyter_http_over_ws
!pip install -q pyngrok

# Jupyter server'ı başlat
import os
from pyngrok import ngrok
import subprocess

# Ngrok token (https://dashboard.ngrok.com/get-started/your-authtoken)
NGROK_TOKEN = "BURAYA_NGROK_TOKEN_GİRİN"
ngrok.set_auth_token(NGROK_TOKEN)

# Jupyter server'ı başlat
os.system('jupyter notebook --NotebookApp.allow_origin="*" --port=8888 --no-browser &')

# Ngrok tunnel aç
public_url = ngrok.connect(8888)
print(f"\n🔗 Jupyter Server URL: {public_url}")
print(f"\n✅ Bu URL'yi VS Code'da kullanın!")
```

### Adım 2: VS Code'da Bağlan
1. Yukarıdaki kodu çalıştırınca URL alacaksınız
2. VS Code'da "Select Kernel" > "Existing Jupyter Server"
3. URL'yi yapıştırın

---

## Yöntem 3: Colab Eklentisi (En Pratik)

### Gerekli Eklenti
```bash
code --install-extension google.colab
```

### Kullanım
1. VS Code'da notebook'u aç
2. Kernel seçerken "Connect to Colab" seçeneği görünecek
3. Google hesabınızla giriş yapın
4. GPU seçin (T4/A100)

---

## 💡 Hızlı Test

VS Code'da bu notebook'u açıkken:
1. `Ctrl+Shift+P` (Command Palette)
2. Şunu yazın: "Notebook: Select Notebook Kernel"
3. "Existing Jupyter Server..." seçin
4. Colab URL'inizi girin

---

## ⚡ Performans İpuçları

- **Colab Pro** kullanıyorsanız: A100 GPU seçin
- **Ücretsiz** kullanıyorsanız: T4 GPU yeterli
- **Session süresi**: 12 saat (Pro'da daha uzun)
- **Bağlantı kesilirse**: Kerneli yeniden seçin

---

## 🔧 Sorun Giderme

### "Connection refused" hatası:
```python
# Colab'da firewall'u devre dışı bırakın
!jupyter notebook --NotebookApp.allow_origin='*' --NotebookApp.disable_check_xsrf=True
```

### Token hatası:
```python
# Token olmadan başlat
!jupyter notebook --NotebookApp.token='' --NotebookApp.password=''
```

---

🎉 **Hazırsınız! Artık VS Code'dan Colab GPU'sunu kullanabilirsiniz!**
