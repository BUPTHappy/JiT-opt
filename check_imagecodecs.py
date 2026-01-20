#!/usr/bin/env python3
"""检查imagecodecs的JPEG-XL支持"""
import sys

print("=" * 60)
print("Checking imagecodecs JPEG-XL support")
print("=" * 60)

try:
    import imagecodecs
    print(f"✓ imagecodecs imported successfully")
    print(f"  Version: {imagecodecs.__version__}")
    print(f"  JPEGXL available: {imagecodecs.JPEGXL}")
    
    # 检查是否是stub
    is_stub = str(imagecodecs.JPEGXL).find('STUB') != -1 or str(type(imagecodecs.JPEGXL)).find('STUB') != -1
    
    if is_stub:
        print("✗ JPEG-XL support is NOT available (STUB detected)")
        print("\nThis means imagecodecs was installed but compiled without JPEG-XL support.")
        print("You need to install libjxl first, then reinstall imagecodecs:")
        print("\nSolution:")
        print("  1. Install libjxl: conda install -c conda-forge libjxl")
        print("  2. Reinstall imagecodecs: conda install -c conda-forge imagecodecs --force-reinstall")
        print("\nOr in one command:")
        print("  conda install -c conda-forge libjxl imagecodecs --force-reinstall")
    elif imagecodecs.JPEGXL:
        print("✓ JPEG-XL support is available!")
        
        # 检查numcodecs注册
        print("\nChecking numcodecs registration...")
        try:
            from numcodecs.registry import get_codec
            codec = get_codec({"id": "imagecodecs_jpegxl"})
            print(f"✓ imagecodecs_jpegxl codec is registered: {codec}")
        except (ValueError, TypeError) as e:
            print(f"✗ imagecodecs_jpegxl codec is NOT registered: {e}")
            print("\nTrying to register it manually...")
            
            # 尝试手动注册
            try:
                from numcodecs.registry import register_codec
                from numcodecs.abc import Codec
                
                class JpegXl(Codec):
                    codec_id = "imagecodecs_jpegxl"
                    
                    def __init__(self, **kwargs):
                        self.kwargs = kwargs
                    
                    def encode(self, buf):
                        return imagecodecs.jpegxl_encode(np.asarray(buf), **self.kwargs)
                    
                    def decode(self, buf, out=None):
                        return imagecodecs.jpegxl_decode(buf, out=out)
                
                import numpy as np
                register_codec(JpegXl)
                
                # 验证
                codec = get_codec({"id": "imagecodecs_jpegxl"})
                print(f"✓ Successfully registered manually: {codec}")
            except Exception as e2:
                print(f"✗ Failed to register manually: {e2}")
                import traceback
                traceback.print_exc()
    else:
        print("✗ JPEG-XL support is NOT available")
        print("\nThis means imagecodecs was installed but without JPEG-XL support.")
        print("You may need to:")
        print("  1. Install libjxl: conda install -c conda-forge libjxl")
        print("  2. Reinstall imagecodecs: conda install -c conda-forge imagecodecs --force-reinstall")
        
except ImportError as e:
    print(f"✗ Failed to import imagecodecs: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
