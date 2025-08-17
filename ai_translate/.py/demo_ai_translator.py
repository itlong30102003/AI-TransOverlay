import cv2
import numpy as np
import mss
import tempfile
import os
import sys
import io
from paddleocr import PaddleOCR
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Fix encoding cho Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Khởi tạo OCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

# Vùng màn hình cần OCR (full HD screen 1920x1080, bạn chỉnh lại nếu khác)
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

class M2M100Translator:
    def __init__(self, model_name="facebook/m2m100_418M"):
        """
        Initialize M2M-100 translator
        Models available:
        - facebook/m2m100_418M (small, fast)
        - facebook/m2m100_1.2B (larger, better quality)
        """
        print(f"Loading M2M-100 model: {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        try:
            # Load model and tokenizer
            self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
            self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            print("✅ M2M-100 model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading M2M-100: {e}")
            print("Make sure you have transformers installed: pip install transformers torch")
            raise
        
        # Language codes for M2M-100
        self.lang_codes = {
            "en": "en",  # English
            "vi": "vi",  # Vietnamese
            "zh": "zh",  # Chinese
            "ja": "ja",  # Japanese
            "ko": "ko",  # Korean
            "fr": "fr",  # French
            "de": "de",  # German
            "es": "es",  # Spanish
        }
    
    def translate_text(self, text, source_lang="en", target_lang="vi"):
        """Translate text using M2M-100"""
        if not text or not text.strip():
            return text
            
        # Skip if text is too short or just numbers/symbols
        if len(text.strip()) < 2 or not any(c.isalpha() for c in text):
            return text
            
        # Auto-detect if it's already Vietnamese
        if not self.is_likely_english(text):
            return text
            
        try:
            # Prepare input
            text = text.strip()
            
            # Set source language
            self.tokenizer.src_lang = source_lang
            
            # Tokenize
            encoded = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **encoded,
                    forced_bos_token_id=self.tokenizer.get_lang_id(target_lang),
                    max_length=512,
                    num_beams=2,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            
            # Decode
            translated = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            # Clean up translation
            translated = self.clean_translation(translated, text)
            
            return translated if translated else text
            
        except Exception as e:
            print(f"M2M-100 translation error: {e}")
            return self.fallback_translate(text)
    
    def is_likely_english(self, text):
        """Check if text is likely English"""
        if not text:
            return False
            
        # Count ASCII alphabetic characters
        ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
        total_alpha = sum(1 for c in text if c.isalpha())
        
        if total_alpha == 0:
            return False
            
        # If > 80% ASCII letters, likely English
        return ascii_alpha / total_alpha > 0.8
    
    def clean_translation(self, translation, original):
        """Clean up M2M-100 translation output"""
        if not translation:
            return original
            
        translation = translation.strip()
        
        # Remove if translation is identical to original
        if translation.lower() == original.lower():
            return self.fallback_translate(original)
            
        # Remove if translation is suspiciously long
        if len(translation) > len(original) * 2.5:
            return self.fallback_translate(original)
            
        return translation
    
    def fallback_translate(self, text):
        """Simple fallback translation"""
        translations = {
            "hello": "xin chào", "world": "thế giới", "good": "tốt",
            "morning": "buổi sáng", "evening": "buổi tối", "night": "đêm",
            "thank you": "cảm ơn", "thanks": "cảm ơn", "please": "xin",
            "apple": "Apple", "macbook": "MacBook", "iphone": "iPhone",
            "discount": "giảm giá", "sale": "khuyến mãi", "price": "giá",
            "buy": "mua", "shop": "cửa hàng", "free": "miễn phí",
            "new": "mới", "hot": "nóng", "menu": "thực đơn",
            "login": "đăng nhập", "password": "mật khẩu", "email": "email",
            "search": "tìm kiếm", "home": "trang chủ", "help": "trợ giúp",
            "settings": "cài đặt", "profile": "hồ sơ", "account": "tài khoản"
        }
        
        text_lower = text.lower().strip()
        
        # Exact match
        if text_lower in translations:
            return translations[text_lower]
            
        # Partial match
        for en, vi in translations.items():
            if en in text_lower:
                return text_lower.replace(en, vi)
                
        return f"[{text}]"  # Return original with brackets if no translation

# Khởi tạo M2M-100 translator
print("Initializing M2M-100 Translator...")
ai_translator = M2M100Translator(model_name="facebook/m2m100_418M")  # Sử dụng model nhỏ, nhanh

def overlay_translate():
    with mss.mss() as sct:
        while True:
            try:
                # Chụp màn hình
                screenshot = sct.grab(monitor)
                
                # Lưu screenshot tạm thời để OCR đọc
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_path = temp_file.name
                    mss.tools.to_png(screenshot.rgb, screenshot.size, output=temp_path)
                
                # OCR predict từ file
                results = ocr.predict(temp_path)
                
                # Xóa file tạm
                os.unlink(temp_path)
                
                # Chuyển screenshot thành frame để hiển thị
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Debug: In ra cấu trúc kết quả
                print(f"Results type: {type(results)}")
                
                # Xử lý kết quả OCR
                if results:
                    for i, res in enumerate(results):
                        print(f"Processing result {i}")
                        
                        # Thử truy cập theo các cách khác nhau
                        ocr_data = None
                        if hasattr(res, 'res'):
                            print("Found res attribute")
                            ocr_data = res.res
                        elif isinstance(res, dict) and 'res' in res:
                            print("Found res key in dict")
                            ocr_data = res['res']
                        elif isinstance(res, dict):
                            print("Direct dict access")
                            ocr_data = res
                        
                        if ocr_data and isinstance(ocr_data, dict):
                            texts = ocr_data.get('rec_texts', [])
                            scores = ocr_data.get('rec_scores', [])
                            boxes = ocr_data.get('rec_boxes', [])
                            
                            print(f"Found {len(texts)} texts")
                            
                            # Vẽ box + dịch
                            for text, confidence, box in zip(texts, scores, boxes):
                                if not text.strip() or confidence < 0.5:
                                    continue
                                
                                try:
                                    # Dịch bằng AI
                                    translated = ai_translator.translate_text(text)
                                    
                                    # Lấy tọa độ từ rec_boxes
                                    x1, y1, x2, y2 = map(int, box)
                                    
                                    # Đảm bảo tọa độ hợp lệ
                                    height, width = frame.shape[:2]
                                    x1, y1 = max(0, x1), max(30, y1)
                                    x2, y2 = min(width, x2), min(height, y2)
                                    
                                    # Vẽ nền cho text dịch
                                    overlay = frame.copy()
                                    cv2.rectangle(overlay, (x1, y1-30), (x2, y1), (0, 0, 0), -1)
                                    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
                                    
                                    # # Vẽ text gốc (màu trắng, nhỏ)
                                    # cv2.putText(frame, text[:30], (x1, y1-18),
                                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                                    
                                    # Vẽ text dịch (màu xanh lá, lớn)
                                    cv2.putText(frame, translated[:50], (x1, y1-5),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                                    
                                    # Debug console
                                    try:
                                        print(f"[OCR] {text} -> {translated}")
                                    except UnicodeEncodeError:
                                        print(f"[OCR] {text} -> <Vietnamese translation>")
                                
                                except Exception as translate_error:
                                    print(f"Translation error: {translate_error}")
                                    continue
                
                # Hiển thị overlay
                cv2.imshow("AI Overlay Translate", frame)
                
                # Thoát bằng phím q
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                    
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting AI Overlay Translator...")
    print("Press 'q' to quit")
    overlay_translate()