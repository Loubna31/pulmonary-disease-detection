// Sidebar toggle
const sidebar       = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
if (sidebarToggle && sidebar) {
  sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
  });
}

// Auto-dismiss alerts after 5 s
document.querySelectorAll('.alert.alert-dismissible').forEach(alert => {
  setTimeout(() => {
    const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
    if (bsAlert) bsAlert.close();
  }, 5000);
});
