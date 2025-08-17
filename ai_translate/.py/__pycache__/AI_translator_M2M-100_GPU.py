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
        Initialize M2M-100 translator optimized for RTX 3050
        """
        print(f"🚀 Loading M2M-100 model: {model_name}")
        print("🎮 Optimizing for RTX 3050 GPU...")
        
        # Force CUDA if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
        else:
            self.device = torch.device("cpu")
            print("⚠️  CUDA not available, using CPU")
        
        try:
            print("1/3 Loading tokenizer...")
            self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            
            print("2/3 Loading model with GPU optimization...")
            self.model = M2M100ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Use FP16 for RTX 3050 speed boost
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            print("3/3 Moving to GPU...")
            if not hasattr(self.model, 'device') or self.model.device != self.device:
                self.model.to(self.device)
            
            self.model.eval()
            
            # Warm up GPU
            print("🔥 Warming up GPU...")
            self._warmup_gpu()
            
            print("🎉 M2M-100 ready with GPU acceleration!")
            
        except Exception as e:
            print(f"❌ Error loading M2M-100: {e}")
            raise
    
    def _warmup_gpu(self):
        """Warm up GPU with dummy translation"""
        try:
            dummy_text = "Hello"
            self.tokenizer.src_lang = "en"
            encoded = self.tokenizer(dummy_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                _ = self.model.generate(
                    **encoded,
                    forced_bos_token_id=self.tokenizer.get_lang_id("vi"),
                    max_length=10,
                    num_beams=1
                )
            print("✅ GPU warmup completed")
        except:
            print("⚠️  GPU warmup failed, but model should still work")
    
    def translate_text(self, text, source_lang="en", target_lang="vi"):
        """Fast GPU-accelerated translation"""
        if not text or not text.strip():
            return ""
            
        text_clean = text.strip()
        
        # Skip very short text
        if len(text_clean) < 2:
            return ""
            
        # Skip if not English (quick check)
        if not self.is_likely_english(text_clean):
            return ""
            
        try:
            # Set source language
            self.tokenizer.src_lang = source_lang
            
            # Tokenize and move to GPU
            encoded = self.tokenizer(
                text_clean, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=100  # Shorter for real-time performance
            ).to(self.device)
            
            # GPU-accelerated generation
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **encoded,
                    forced_bos_token_id=self.tokenizer.get_lang_id(target_lang),
                    max_length=120,
                    num_beams=1,        # Faster single beam
                    do_sample=False,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True      # Use KV cache for speed
                )
            
            # Decode result
            translated = self.tokenizer.batch_decode(
                generated_tokens, 
                skip_special_tokens=True
            )[0].strip()
            
            # Return only if translation is different and valid
            if translated and translated.lower() != text_clean.lower():
                return translated
            else:
                return self.fallback_translate(text_clean)
                
        except Exception as e:
            print(f"🔧 Translation error: {e}")
            return self.fallback_translate(text_clean)
    
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

# Khởi tạo M2M-100 translator - Thử model nhỏ hơn nếu bị hang
print("Initializing M2M-100 Translator...")
try:
    # Thử model nhỏ nhất trước
    ai_translator = M2M100Translator(model_name="facebook/m2m100_418M")
except Exception as e:
    print(f"❌ M2M-100 failed: {e}")
    print("🔄 Falling back to basic translator...")
    
    # Fallback translator nếu M2M-100 fail
    class BasicTranslator:
        def translate_text(self, text):
            translations = {
                "hello": "xin chào", "world": "thế giới", "good": "tốt",
                "morning": "buổi sáng", "thank you": "cảm ơn", 
                "apple": "Apple", "macbook": "MacBook"
            }
            text_lower = text.lower()
            for en, vi in translations.items():
                if en in text_lower:
                    return text_lower.replace(en, vi)
            return f"[{text}]"
    
    ai_translator = BasicTranslator()

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
                                    
                                    # Chỉ hiển thị nếu có bản dịch
                                    if not translated or translated == text or translated.startswith('['):
                                        continue
                                    
                                    # Lấy tọa độ từ rec_boxes
                                    x1, y1, x2, y2 = map(int, box)
                                    
                                    # Đảm bảo tọa độ hợp lệ
                                    height, width = frame.shape[:2]
                                    x1, y1 = max(0, x1), max(25, y1)
                                    x2, y2 = min(width, x2), min(height, y2)
                                    
                                    # Vẽ nền đen mờ cho text dịch
                                    overlay = frame.copy()
                                    
                                    # Tính toán kích thước text để vẽ nền vừa đủ
                                    font_scale = 0.7
                                    thickness = 2
                                    (text_width, text_height), baseline = cv2.getTextSize(
                                        translated[:60], cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                                    )
                                    
                                    # Vẽ nền đen với padding
                                    padding = 5
                                    bg_x1 = x1 - padding
                                    bg_y1 = y1 - text_height - baseline - padding
                                    bg_x2 = x1 + text_width + padding
                                    bg_y2 = y1 + padding
                                    
                                    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                                    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
                                    
                                    # Vẽ CHỈ text dịch tiếng Việt (màu xanh lá cây sáng)
                                    cv2.putText(frame, translated[:60], (x1, y1-baseline),
                                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 100), thickness)
                                    
                                    # Debug console - chỉ in nếu dịch thành công
                                    try:
                                        print(f"✅ {text} → {translated}")
                                    except UnicodeEncodeError:
                                        print(f"✅ Translation successful")
                                
                                except Exception as translate_error:
                                    # Im lặng, không in lỗi để tránh spam console
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
    print("🚀 Starting GPU-Accelerated AI Overlay Translator...")
    print("🎮 Optimized for RTX 3050")
    print("🇻🇳 Vietnamese-only translation overlay")
    print("⚡ Press 'q' to quit")
    print("=" * 50)
    overlay_translate()