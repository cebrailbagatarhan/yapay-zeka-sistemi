# 🚀 GitHub'a Yükleme Adımları

## Adım 1: GitHub'da Repository Oluşturun

1. GitHub'a gidin: https://github.com
2. Giriş yapın
3. Sağ üstteki **+** butonuna tıklayın
4. **New repository** seçin
5. Repository adı: `yapay-zeka-sistemi` (veya istediğiniz isim)
6. Açıklama: "Gelişmiş AI sistemi - Derin web araştırma + Model eğitimi"
7. Public/Private seçin
8. ❌ "Initialize with README" işaretlemeyin (zaten var)
9. **Create repository** tıklayın

## Adım 2: GitHub Repository URL'sini Kopyalayın

GitHub sayfasında göreceğiniz URL'yi kopyalayın:
```
https://github.com/cebrailbagatarhan/yapay-zeka-sistemi.git

```

## Adım 3: Git Remote Ekleyin ve Push Yapın

PowerShell'de bu komutları çalıştırın:

```powershell
# Remote ekle (URL'yi kendi repository URL'inizle değiştirin)
git remote add origin git remote add origin https://github.com/cebrailbagatarhan/yapay-zeka-sistemi.git


# Branch ismini main yap (modern GitHub standartı)
git branch -M main

# GitHub'a yükle
git push -u origin main
```

### Eğer GitHub şifre isterse:

**Personal Access Token (PAT) kullanmanız gerekiyor:**

1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. **Generate new token (classic)** tıklayın
3. Note: "Yapay Zeka Projesi"
4. **repo** checkbox'ını işaretleyin
5. **Generate token** tıklayın
6. Token'ı kopyalayın (bir daha göremezsiniz!)
7. Git push yaparken şifre yerine bu token'ı kullanın

## Adım 4: Doğrulama

GitHub repository sayfanızı yenileyin, tüm dosyalarınız orada olmalı!

---

## 📊 Projenizde Neler Var?

✅ **35 dosya yüklendi:**
- ✅ Ana uygulama (main.py)
- ✅ 9 modül (src/ klasöründe)
- ✅ 4 Jupyter notebook
- ✅ Derin web araştırma modülü
- ✅ Model eğitim sistemi
- ✅ Test dosyaları
- ✅ README.md
- ✅ requirements.txt
- ✅ .gitignore (büyük dosyalar hariç)

## 🎯 Gelecek Güncellemeler İçin:

```powershell
# Değişiklikleri kaydet
git add .
git commit -m "Özellik: Yeni özellik eklendi"
git push
```

## 🌟 İpuçları:

- Branch kullanarak özellik geliştirin
- Her önemli değişiklik için commit yapın
- README.md'yi güncel tutun
- Issues kullanarak hataları takip edin
- Pull Request ile katkı alın

---

**Hazır! Artık projeniz GitHub'da! 🎉**
