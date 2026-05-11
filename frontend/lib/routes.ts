export function doctorGatewayPath() {
  return "/doctor";
}

export function doctorHomePath(doctorId: string) {
  return `/doctor/${doctorId}`;
}

export function doctorWorkspacePath(doctorId: string) {
  return `/doctor/${doctorId}/workspace`;
}

export function doctorResearchPath(doctorId: string) {
  return `/doctor/${doctorId}/research`;
}

/** Doctor's Corner / MH clinician forum (same route as research). */
export function doctorForumPath(doctorId: string) {
  return doctorResearchPath(doctorId);
}

export function doctorPatientChartPath(doctorId: string, patientId: string) {
  return `/doctor/${doctorId}/patient/${patientId}`;
}

export function doctorPatientNancyConsolePath(doctorId: string, patientId: string) {
  return `${doctorPatientChartPath(doctorId, patientId)}/nancy`;
}

export function doctorPatientSessionPath(doctorId: string, patientId: string) {
  return `${doctorPatientChartPath(doctorId, patientId)}/session`;
}

export function patientPortalHomePath(doctorId: string, patientId: string) {
  return `${doctorPatientChartPath(doctorId, patientId)}/companion`;
}

export function patientPortalNancyPath(doctorId: string, patientId: string) {
  return `${patientPortalHomePath(doctorId, patientId)}/nancy`;
}

export function patientGatewayPath() {
  return "/patient";
}
