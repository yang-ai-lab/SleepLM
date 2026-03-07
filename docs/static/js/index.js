document.addEventListener('DOMContentLoaded', function () {

  var panels = document.querySelectorAll('.tab-panel');
  var navItems = document.querySelectorAll('.nav-links a[data-tab]');
  var brand = document.querySelector('.nav-brand[data-tab]');
  var toggle = document.querySelector('.nav-toggle');
  var navLinks = document.querySelector('.nav-links');

  function switchTab(tabId) {
    panels.forEach(function (p) {
      p.classList.remove('active');
    });
    var target = document.getElementById('tab-' + tabId);
    if (target) {
      target.classList.add('active');
    }

    navItems.forEach(function (a) {
      a.classList.remove('active');
      if (a.getAttribute('data-tab') === tabId) {
        a.classList.add('active');
      }
    });

    window.scrollTo({ top: 0, behavior: 'instant' });

    history.replaceState(null, '', '#' + tabId);
  }

  navItems.forEach(function (a) {
    a.addEventListener('click', function (e) {
      e.preventDefault();
      switchTab(this.getAttribute('data-tab'));
      if (navLinks) navLinks.classList.remove('open');
    });
  });

  if (brand) {
    brand.addEventListener('click', function (e) {
      e.preventDefault();
      switchTab('home');
    });
  }

  if (toggle && navLinks) {
    toggle.addEventListener('click', function () {
      navLinks.classList.toggle('open');
    });
  }

  // Click-based dropdown toggle
  var dropdownTriggers = document.querySelectorAll('.dropdown-trigger');
  dropdownTriggers.forEach(function (trigger) {
    trigger.addEventListener('click', function (e) {
      e.preventDefault();
      e.stopPropagation();
      var parent = this.closest('.nav-dropdown');
      var wasOpen = parent.classList.contains('open');
      // Close all dropdowns first
      document.querySelectorAll('.nav-dropdown.open').forEach(function (d) {
        d.classList.remove('open');
      });
      if (!wasOpen) parent.classList.add('open');
    });
  });

  // Close dropdown when clicking anywhere else
  document.addEventListener('click', function () {
    document.querySelectorAll('.nav-dropdown.open').forEach(function (d) {
      d.classList.remove('open');
    });
  });

  var hash = window.location.hash.replace('#', '');
  var validTabs = ['home', 'architecture', 'results', 'citation'];
  if (hash && validTabs.indexOf(hash) !== -1) {
    switchTab(hash);
  }

});
