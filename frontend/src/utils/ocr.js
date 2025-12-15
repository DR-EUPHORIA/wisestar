export const performOCR = async (imageFile) => {
  try {
    const Tesseract = await import('tesseract.js');
    
    const { data: { text } } = await Tesseract.recognize(
      imageFile,
      'chi_sim+eng',
      {
        logger: (m) => {
          if (m.status === 'recognizing text') {
            console.log(`OCR Progress: ${Math.round(m.progress * 100)}%`);
          }
        }
      }
    );
    
    return {
      success: true,
      text: text.trim(),
      error: null
    };
  } catch (error) {
    console.error('OCR Error:', error);
    return {
      success: false,
      text: '',
      error: error.message
    };
  }
};

export const extractMathProblem = (ocrText) => {
  const lines = ocrText.split('\n').filter(line => line.trim());
  
  const problemPatterns = [
    /[0-9]+[\.、]\s*(.+)/,
    /题目[:：]\s*(.+)/i,
    /问题[:：]\s*(.+)/i
  ];
  
  for (const pattern of problemPatterns) {
    const match = ocrText.match(pattern);
    if (match) {
      return match[1].trim();
    }
  }
  
  return lines.join(' ');
};
