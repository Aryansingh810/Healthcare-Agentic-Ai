async function requireAuth(redirect = "/login.html") {
  try {
    const s = await getSession();
    if (!s.logged_in) {
      window.location.href = redirect;
      return null;
    }
    return s;
  } catch {
    window.location.href = redirect;
    return null;
  }
}

function requireRole(s, role, redirect = "/dashboard.html") {
  if (!s || s.role !== role) {
    alert(`This area is for ${role} accounts.`);
    window.location.href = redirect;
    return false;
  }
  return true;
}
