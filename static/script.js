// Handle file uploads and previews
function handleSingleFileSelect(input, preview) {
  const file = input.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function(e) {
    const isVideo = file.type.startsWith('video/');
    const isImage = file.type.startsWith('image/');

    let previewContent = `
        <button class="remove-file" onclick="removeFile('${input.id}')">
          <i class="fas fa-times"></i>
        </button>`;

    if (isVideo) {
      previewContent += `<video src="${e.target.result}" controls class="preview-image"></video>`;
    } else if (isImage) {
      previewContent += `<img src="${e.target.result}" alt="Preview" class="preview-image">`;
    }

    previewContent += `
        <div class="file-info single">
          <i class="fas fa-file-${isVideo ? 'video' : 'image'} file-icon"></i>
          <span class="file-name">${file.name}</span>
        </div>
      `;

    preview.innerHTML = previewContent;
    preview.style.display = 'block';
    updateSwapButton();
  };
  reader.readAsDataURL(file);
}

function handleMultipleFilesSelect(input, preview) {
  const files = Array.from(input.files);
  if (files.length === 0) return;

  let previewContent = `<button class="remove-file" onclick="removeFile('${input.id}')">
    <i class="fas fa-times"></i>
  </button>`;

  files.forEach((file, index) => {
    const reader = new FileReader();
    reader.onload = function(e) {
      const isVideo = file.type.startsWith('video/');
      const isImage = file.type.startsWith('image/');

      previewContent += `
          <div class="file-item">
            ${isVideo ?
              `<video src="${e.target.result}" class="preview-image"></video>` :
              `<img src="${e.target.result}" alt="${file.name}" class="preview-image">`
            }
            <div class="file-info">
              <button class="remove-single-file" onclick="removeSingleFile('${input.id}', '${file.name}')">
                <i class="fas fa-times"></i>
              </button>
              <i class="fas fa-file-${isVideo ? 'video' : 'image'} file-icon"></i>
              <span class="file-name">${file.name}</span>
            </div>
          </div>
        `;

      // Show preview after all files are loaded
      if (index === files.length - 1) {
        preview.innerHTML = previewContent;
        preview.style.display = 'block';
        updateSwapButton();
      }
    };
    reader.readAsDataURL(file);
  });
}

function handleFileSelect(input, preview) {
  const isMultiple = input.multiple;
  if (isMultiple) {
    handleMultipleFilesSelect(input, preview);
  } else {
    handleSingleFileSelect(input, preview);
  }
}

function removeFile(inputId) {
  const input = document.getElementById(inputId);
  const container = input.closest('.upload-container');
  const preview = container.querySelector('.file-preview');

  input.value = '';
  preview.style.display = 'none';
  preview.innerHTML = '';
  updateSwapButton();
}

function updateSwapButton() {
  const sourceInput = document.getElementById('sourceInput');
  const targetsInput = document.getElementById('targetsInput');
  const button = document.getElementById('swapButton');

  if (sourceInput.files[0] && targetsInput.files.length > 0) {
    button.disabled = false;
    button.innerHTML = `<i class="fas fa-magic"></i> Swap ${targetsInput.files.length} Files!`;
  } else {
    button.disabled = true;
    button.innerHTML = 'Chọn files trước';
  }
}

// Drag and drop functionality
function handleDragOver(e) {
  e.preventDefault();
  e.stopPropagation();
  this.classList.add('dragover');
}

function handleDragLeave(e) {
  e.preventDefault();
  e.stopPropagation();
  this.classList.remove('dragover');
}

function handleDrop(e, inputId) {
  e.preventDefault();
  e.stopPropagation();
  this.classList.remove('dragover');
  console.log('Drop event fired for', inputId);

  const files = e.dataTransfer.files;
  console.log('Files dropped:', files.length);
  if (files.length > 0) {
    const input = document.getElementById(inputId);
    console.log('Input found:', input);
    console.log('Input.multiple:', input.multiple);

    // For multiple files, clear existing files first and append new ones
    if (input.multiple) {
      // Get existing files
      const existingFiles = Array.from(input.files || []);
      console.log('Existing files:', existingFiles.length);

      // Combine with new files
      const allFiles = [...existingFiles, ...Array.from(files)];
      console.log('All files combined:', allFiles.length);

      // Create new DataTransfer
      const dt = new DataTransfer();
      allFiles.forEach(file => {
        dt.items.add(file);
        console.log('Added file to DataTransfer:', file.name);
      });

      input.files = dt.files;
    } else {
      // For single file input, just set the first file
      const dt = new DataTransfer();
      dt.items.add(files[0]);
      input.files = dt.files;
    }

    console.log('Final input.files length:', input.files.length);

    // Trigger change event manually
    const changeEvent = new Event('change', { bubbles: true });
    input.dispatchEvent(changeEvent);
  }
}

// Event listeners
document.getElementById('sourceInput').addEventListener('change', function(e) {
  const preview = document.getElementById('sourcePreview');
  handleFileSelect(e.target, preview);
});

document.getElementById('targetsInput').addEventListener('change', function(e) {
  const preview = document.getElementById('targetsPreview');
  handleFileSelect(e.target, preview);
});

// Drag and drop listeners
document.getElementById('sourceContainer').addEventListener('dragover', handleDragOver);
document.getElementById('sourceContainer').addEventListener('dragleave', handleDragLeave);
document.getElementById('sourceContainer').addEventListener('drop', function(e) { handleDrop.call(this, e, 'sourceInput'); });

document.getElementById('targetsContainer').addEventListener('dragover', handleDragOver);
document.getElementById('targetsContainer').addEventListener('dragleave', handleDragLeave);
document.getElementById('targetsContainer').addEventListener('drop', function(e) { handleDrop.call(this, e, 'targetsInput'); });

// Modal functions
function showFullImage(imageSrc, caption) {
  const modal = document.getElementById('imageModal');
  const modalImage = document.getElementById('modalImage');
  const modalCaption = document.getElementById('modalCaption');

  modalImage.src = imageSrc;
  modalCaption.textContent = caption;
  modal.style.display = 'flex';
}

function closeModal() {
  const modal = document.getElementById('imageModal');
  modal.style.display = 'none';
}

function downloadFile(imageSrc, filename) {
  // Extract original filename without UUID
  const originalFilename = filename;

  // Create a temporary anchor element
  const link = document.createElement('a');
  link.href = imageSrc;
  link.download = originalFilename;

  // Append to body, click, and remove
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Remove single file function
function removeSingleFile(inputId, filename) {
  const input = document.getElementById(inputId);
  const files = Array.from(input.files);

  // Filter out the file to remove
  const filteredFiles = files.filter(file => file.name !== filename);

  // Update input files
  const dt = new DataTransfer();
  filteredFiles.forEach(file => {
    dt.items.add(file);
  });
  input.files = dt.files;

  // Re-trigger preview update
  const changeEvent = new Event('change', { bubbles: true });
  input.dispatchEvent(changeEvent);
}

// Form submission
const form = document.getElementById("swapForm");
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const button = document.getElementById('swapButton');
  const resultDiv = document.getElementById("result");
  const progressDiv = document.getElementById("progress");
  const progressFill = document.getElementById("progressFill");
  const progressText = document.getElementById("progressText");

  // Reset UI
  resultDiv.style.display = 'none';
  resultDiv.innerHTML = '';

  // Show progress
  progressDiv.style.display = 'block';
  button.disabled = true;
  button.innerHTML = '<span class="loading-spinner"></span> Đang Xử Lý...';

  // Simulate progress
  let progress = 0;
  const progressInterval = setInterval(() => {
    progress += Math.random() * 15;
    if (progress > 90) progress = 90;
    progressFill.style.width = progress + '%';
    progressText.textContent = 'Đang xử lý... ' + Math.round(progress) + '%';
  }, 200);

  try {
    const formData = new FormData(form);
    const res = await fetch("/swapface", { method: "POST", body: formData });
    const data = await res.json();

    clearInterval(progressInterval);
    progressFill.style.width = '100%';
    progressText.textContent = 'Hoàn thành!';

    setTimeout(() => {
      progressDiv.style.display = 'none';
      progressFill.style.width = '0%';

      if (data.results && data.results.length > 0) {
        const successResults = data.results.filter(r => r.result);
        const errorResults = data.results.filter(r => r.error);

        if (successResults.length > 0) {
          resultDiv.innerHTML = `
            <div class="result-container">
              <div class="result-success">
                <h3><i class="fas fa-check-circle"></i> Batch Swap Thành Công!</h3>
                <p>${successResults.length}/${data.results.length} files được xử lý thành công</p>
              </div>
                  <div class="files-preview results-grid" style="grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); margin-top: 2rem;">${
                    successResults.map((result, index) => `
                      <div class="file-item">
                        <img src="${result.result}" alt="${result.original_name}" style="width: 100%; height: 350px; object-fit: cover; border-radius: 0.75rem; cursor: pointer;" onclick="showFullImage('${result.result}', '${result.original_name}')">
                        <div class="file-info" style="padding: 0.75rem; display: flex; flex-direction: column; gap: 0.5rem;">
                          <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                            <i class="fas fa-file-${result.type === 'image' ? 'image' : 'video'} file-icon"></i>
                            <span class="file-name">${result.original_name}</span>
                          </div>
                          <div style="display: flex; gap: 0.5rem; justify-content: center;">
                            <button class="download-btn" onclick="downloadFile('${result.result}', '${result.original_name}')" style="background: var(--success); color: white; border: none; padding: 0.5rem 1rem; border-radius: 0.5rem; cursor: pointer; font-size: 0.875rem;">
                              <i class="fas fa-download"></i> Tải về
                            </button>
                          </div>
                        </div>
                      </div>
                    `).join('')
                  }</div>

                  <!-- Modal for full image view -->
                  <div id="imageModal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000; align-items: center; justify-content: center;" onclick="closeModal()">
                    <div style="background: white; padding: 2rem; border-radius: 1rem; max-width: 90%; max-height: 90%; position: relative; overflow: hidden;" onclick="event.stopPropagation()">
                      <button onclick="closeModal()" style="position: absolute; top: 1rem; right: 1rem; background: var(--error); color: white; border: none; border-radius: 50%; width: 3rem; height: 3rem; font-size: 1.5rem; cursor: pointer;">×</button>
                      <img id="modalImage" src="" alt="" style="max-width: 100%; max-height: 100%; object-fit: contain;">
                      <p id="modalCaption" style="text-align: center; margin-top: 1rem; font-weight: 500;"></p>
                    </div>
                  </div>
            </div>
          `;

          if (errorResults.length > 0) {
            resultDiv.innerHTML += `
              <div class="result-container" style="margin-top: 2rem;">
                <div class="error-message">
                  <h3><i class="fas fa-exclamation-triangle"></i> Một số files thất bại</h3>
                  <ul style="text-align: left; margin-top: 1rem;">
                    ${errorResults.map(result => `<li>${result.original_name || 'Unknown'}: ${result.error}</li>`).join('')}
                  </ul>
                </div>
              </div>
            `;
          }
        } else {
          resultDiv.innerHTML = `
            <div class="result-container">
              <div class="error-message">
                <h3><i class="fas fa-times-circle"></i> Tất cả files thất bại</h3>
                <ul style="text-align: left; margin-top: 1rem;">
                  ${errorResults.map(result => `<li>${result.original_name || 'Unknown'}: ${result.error}</li>`).join('')}
                </ul>
              </div>
            </div>
          `;
        }
      } else {
        resultDiv.innerHTML = `
          <div class="result-container">
            <div class="error-message">
              <h3><i class="fas fa-exclamation-triangle"></i> Có Lỗi Xảy Ra</h3>
              <p>${data.error || 'Vui lòng thử lại sau'}</p>
            </div>
          </div>
        `;
      }

      resultDiv.style.display = 'block';

      // Reset button
      button.disabled = false;
      button.innerHTML = '<i class="fas fa-magic"></i> Swap Ngay!';
    }, 500);

  } catch (error) {
    clearInterval(progressInterval);
    progressDiv.style.display = 'none';
    progressFill.style.width = '0%';

    resultDiv.innerHTML = `
      <div class="result-container">
        <div class="error-message">
          <h3><i class="fas fa-exclamation-triangle"></i> Lỗi Kết Nối</h3>
          <p>Vui lòng kiểm tra kết nối mạng và thử lại</p>
        </div>
      </div>
    `;
    resultDiv.style.display = 'block';

    // Reset button
    button.disabled = false;
    button.innerHTML = '<i class="fas fa-magic"></i> Swap Ngay!';
  }
});
