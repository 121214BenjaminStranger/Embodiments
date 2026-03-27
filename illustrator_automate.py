import os
import sys
import time
import win32com.client

# Robust Illustrator automation: looks for HALLEY_portrait.png in common locations,
# verifies Illustrator COM ProgID registration, dispatches the app, places the
# image into a new document, and saves as HALLEY_portrait.ai.

SEARCH_PATHS = [
    os.path.dirname(os.path.abspath(__file__)),
    os.getcwd(),
    os.path.expanduser('~'),
    'C:\\Users\\benja',
]


def find_image():
    names = ['HALLEY_portrait.png', 'HALLEY_portrait_v5_enhanced.png', 'HALLEY_portrait_v5_fullres.png', 'HALLEY_portrait.png']
    for base in SEARCH_PATHS:
        for n in names:
            p = os.path.join(base, n)
            if os.path.exists(p):
                return p
    return None


def illustrator_registered():
    try:
        import winreg
        candidates = [
            'Illustrator.Application',
            'Illustrator.Application.1',
        ]
        for k in candidates:
            try:
                with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, k):
                    return True
            except FileNotFoundError:
                continue
        return False
    except Exception:
        return False


def main():
    img = find_image()
    if not img:
        print('ERROR: Could not find HALLEY portrait image in:', SEARCH_PATHS)
        sys.exit(2)
    print('Found image:', img)

    if not illustrator_registered():
        print('Warning: Illustrator COM ProgID not found in registry. Proceeding anyway (you said Illustrator is already open).')

    try:
        print('Dispatching Illustrator...')
        ai = win32com.client.Dispatch('Illustrator.Application')
        try:
            ai.Visible = True
        except Exception:
            # Some Illustrator COM builds do not allow setting Visible; ignore.
            pass
    except Exception as e:
        print('Failed to dispatch Illustrator COM object:', e)
        sys.exit(5)

    # Try to create a new document and place the image
    try:
        doc = ai.Documents.Add()
        # Place image
        try:
            placed = doc.PlacedItems.Add()
            # Some versions expect a 'File' attribute or SetPlacedFile method
            try:
                placed.File = img
            except Exception:
                try:
                    placed.FileName = img
                except Exception:
                    # fallback: use the Documents.Open method on a raster file
                    try:
                        doc.Close()
                        doc = ai.Open(img)
                    except Exception as e2:
                        print('Could not place or open image programmatically:', e2)
                        raise
        except Exception as e:
            print('PlacedItems.Add failed, trying Documents.Open fallback:', e)
            try:
                doc = ai.Open(img)
            except Exception as e2:
                print('Open fallback failed:', e2)
                raise

        # Give Illustrator a moment to process
        time.sleep(1)

        out_ai = os.path.join(os.path.dirname(img), 'HALLEY_portrait.ai')
        print('Saving AI to', out_ai)
        try:
            # Try SaveAs; API may require an options object — simplest attempt first
            doc.SaveAs(out_ai)
            print('Saved:', out_ai)
        except Exception as e:
            print('SaveAs failed:', e)

        try:
            doc.Close()
        except Exception:
            pass

    except Exception as e_main:
        print('Automation failed:', e_main)
        sys.exit(6)


if __name__ == '__main__':
    main()
