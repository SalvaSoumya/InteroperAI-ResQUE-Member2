import React, { useEffect, useState } from "react";

// Configurable API base URL with local development fallback
const API = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8001";

const navigation = [
  ["dashboard", "Dashboard"],
  ["incidents", "Incident Assessment"],
  ["shelters", "Shelter Management"],
  ["resources", "Resource Management"],
  ["analytics", "Analytics"],
  ["recommendations", "Recommendations"],
];

function Badge({ text, type }) {
  const norm = String(text || "unknown").toLowerCase().trim().replace(/[\s\/_]+/g, "-");
  return (
    <span className={`badge ${norm} ${type ? `badge-${type}` : ""}`}>
      {text}
    </span>
  );
}

function StatCard({ label, value, subtext, danger, warning, icon = "●" }) {
  return (
    <div className="stat-card">
      <div className={`stat-icon ${danger ? "danger" : warning ? "warning" : ""}`}>{icon}</div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <small>{label}</small>
        <strong>{value}</strong>
        {subtext && <span className="stat-subtext">{subtext}</span>}
      </div>
    </div>
  );
}

function buildIncidentRecommendations(incidents, shelters = [], resources = []) {
  const actionByType = {
    flood: "Verify local water levels, identify exposed communities, and coordinate evacuation support where required.",
    wildfire: "Confirm perimeter, alert local emergency responders, and verify active evacuation routes.",
    fire: "Confirm the affected area, alert local emergency responders, and check whether evacuation support is needed.",
    earthquake: "Check for structural damage, identify people needing assistance, and coordinate a local safety assessment.",
    cyclone: "Monitor official weather alerts, verify exposed locations, and prepare emergency communication and shelter coordination.",
    hurricane: "Coordinate storm surge warning alerts, stage high-water rescue equipment, and prepare regional intake centers.",
    tornado: "Deploy search and rescue teams to mapped damage path; establish field triage posts.",
    landslide: "Check slope and road safety, identify potentially isolated communities, and coordinate a local hazard assessment.",
    drought: "Review water availability in the affected area and coordinate needs assessment for essential supplies.",
    heatwave: "Check for vulnerable people, communicate heat-safety guidance, and coordinate access to drinking water.",
    storm: "Check for weather-related hazards, blocked routes, and immediate assistance needs.",
    "rip current": "Broadcast coastal surf advisories, station lifeguard units, and verify hospital emergency admission readiness.",
  };

  return [...incidents]
    .sort((a, b) => {
      const aCrit = (a.impact_tier || "").includes("Critical") ? 2 : 0;
      const bCrit = (b.impact_tier || "").includes("Critical") ? 2 : 0;
      return (bCrit + (Number(b.affected) || 0)) - (aCrit + (Number(a.affected) || 0));
    })
    .slice(0, 20)
    .map((incident) => {
      const type = String(incident.type || "").toLowerCase();
      const matchedKey = Object.keys(actionByType).find((key) => type.includes(key));
      const affected = Number(incident.affected) || 0;
      const casualties = incident.reported_casualties ?? null;
      const injuries = incident.injured_population ?? null;
      const loss = incident.estimated_loss ?? null;
      const hasCasualtyEvidence = casualties != null && !isNaN(casualties);
      const hasInjuryEvidence = injuries != null && !isNaN(injuries);
      const hasLossEvidence = loss != null && !isNaN(loss);

      let evidence = "Impact telemetry unrecorded in preliminary report.";
      let guidanceType = "Field Verification Request";
      let reason = "Missing casualty/damage evidence requires on-site verification before dispatching specialized relief.";

      if (hasCasualtyEvidence && Number(casualties) > 0) {
        evidence = `${casualties} verified fatality(ies) reported${hasInjuryEvidence && Number(injuries) > 0 ? `, ${injuries} injured persons` : ""}.`;
        guidanceType = "Rule-Based Impact Guidance";
        reason = "Confirmed casualty presence requires immediate search, rescue, and medical emergency teams.";
      } else if (hasLossEvidence && Number(loss) >= 5000000) {
        evidence = `$${Number(loss).toLocaleString()} estimated property/crop damage (Critical Tier).`;
        guidanceType = "Rule-Based Impact Guidance";
        reason = "Severe structural/economic loss requires rapid engineering inspection and shelter mobilization.";
      } else if (hasLossEvidence && Number(loss) > 0) {
        evidence = `$${Number(loss).toLocaleString()} estimated property/crop damage.`;
        guidanceType = "Rule-Based Impact Guidance";
        reason = "Moderate structural/crop losses require localized safety inspection and relief supply delivery.";
      } else if (incident.is_anomaly || incident.anomaly_status === "Anomaly") {
        evidence = `Isolation Forest multidimensional statistical outlier (score: ${Number(incident.anomaly_score || 0).toFixed(4)}).`;
        guidanceType = "Statistical Anomaly Review";
        reason = "Unusual multidimensional feature pattern identified; requires human analyst verification rather than automatic emergency declaration.";
      } else if (hasCasualtyEvidence && Number(casualties) === 0 && hasLossEvidence && Number(loss) === 0) {
        evidence = "Verified zero casualties and $0 reported loss.";
        guidanceType = "Operational Baseline Guidance";
        reason = "All impact dimensions verified negative; continue standard observational monitoring.";
      }

      // Check for matching shelter & resource inventory
      const incZone = String(incident.zone || incident.incident_location || "").toLowerCase();
      const matchedShelter = shelters.find((s) => {
        const sLoc = String(s.location || "").toLowerCase();
        return sLoc && (incZone.includes(sLoc) || sLoc.includes(incZone));
      });
      const matchedResource = resources.find((r) => {
        const rLoc = String(r.location || "").toLowerCase();
        return rLoc && (incZone.includes(rLoc) || rLoc.includes(incZone));
      });

      let inventoryNote = "";
      if (matchedShelter) {
        if (matchedShelter.status === "Full" || matchedShelter.occupancy >= matchedShelter.capacity) {
          inventoryNote += ` [GUARDRAIL WARNING: Nearby shelter '${matchedShelter.shelter_name}' is at 100% CAPACITY (${matchedShelter.occupancy}/${matchedShelter.capacity}). DO NOT route evacuees here.]`;
        } else {
          inventoryNote += ` [Nearby Shelter: '${matchedShelter.shelter_name}' (${matchedShelter.capacity - matchedShelter.occupancy} beds available, ${matchedShelter.utilization_percentage}% utilized).]`;
        }
      }
      if (matchedResource && matchedResource.shortage > 0) {
        inventoryNote += ` [Supply Alert: ${matchedResource.resource_type} deficit of ${matchedResource.shortage} ${matchedResource.unit} at ${matchedResource.location}.]`;
      }

      const defaultAction = matchedKey
        ? actionByType[matchedKey]
        : "Verify incident details with authorized field responders and assess immediate emergency assistance requirements.";
      const action = `${defaultAction}${affected > 0 ? ` Reported exposed population: ${affected.toLocaleString()}.` : ""}${inventoryNote}`;

      return {
        incident_id: incident.id,
        zone: incident.zone || "Location not provided",
        location: incident.zone || "Location not provided",
        disaster_type: incident.type || "Incident",
        priority: incident.priority || "Unknown",
        impact_tier: incident.impact_tier || "Unassessed",
        guidance_type: guidanceType,
        supporting_evidence: evidence,
        action: action,
        recommended_action: action,
        reason: reason,
        source: "integrated",
        matched_shelter: matchedShelter || null,
        matched_resource: matchedResource || null
      };
    });
}

function App() {
  const [data, setData] = useState(null);
  const [processedData, setProcessedData] = useState(null);
  const [activityLogs, setActivityLogs] = useState([]);
  const [shelters, setShelters] = useState([]);
  const [shelterSummary, setShelterSummary] = useState(null);
  const [resources, setResources] = useState([]);
  const [resourceSummary, setResourceSummary] = useState(null);
  const [page, setPage] = useState("dashboard");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);

  // Incident-to-Inventory match modal state
  const [matchingIncident, setMatchingIncident] = useState(null);
  const [inventoryMatchResult, setInventoryMatchResult] = useState(null);
  const [matchLoading, setMatchLoading] = useState(false);
  const [matchError, setMatchError] = useState("");

  async function loadShelters() {
    try {
      const res = await fetch(`${API}/api/shelters`);
      if (res.ok) {
        const json = await res.json();
        setShelters(json.shelters || []);
        setShelterSummary(json.summary || null);
      }
    } catch {
      // Keep existing shelters if network fails
    }
  }

  async function loadResources() {
    try {
      const res = await fetch(`${API}/api/resources`);
      if (res.ok) {
        const json = await res.json();
        setResources(json.resources || []);
        setResourceSummary(json.summary || null);
      }
    } catch {
      // Keep existing resources if network fails
    }
  }

  async function loadData() {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API}/api/assess`);

      if (!response.ok) {
        throw new Error("Backend assessment request failed");
      }

      const result = await response.json();
      setData(result);

      // Load the latest processed CSV received from Member 1.
      try {
        const processedResponse = await fetch(`${API}/api/processed-results`);
        if (processedResponse.ok) {
          const processedResult = await processedResponse.json();
          setProcessedData(processedResult);
        } else {
          setProcessedData(null);
        }
      } catch {
        setProcessedData(null);
      }

      // Load recent API activity logs for audit and traceability
      try {
        const logsResponse = await fetch(`${API}/api/activity-logs?limit=5`);
        if (logsResponse.ok) {
          const logsResult = await logsResponse.json();
          setActivityLogs(logsResult.logs || []);
        } else {
          setActivityLogs([]);
        }
      } catch {
        setActivityLogs([]);
      }

      // Load persistent shelters & resources
      await Promise.all([loadShelters(), loadResources()]);

    } catch (err) {
      setError("Unable to connect to the backend. Please make sure FastAPI is running at " + API);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadData();
  }, []);

  async function handleOpenInventoryMatch(incident) {
    setMatchingIncident(incident);
    setInventoryMatchResult(null);
    setMatchError("");
    setMatchLoading(true);

    try {
      const loc = encodeURIComponent(incident.zone || incident.incident_location || "");
      const res = await fetch(`${API}/api/inventory/match?location=${loc}&incident_id=${encodeURIComponent(incident.id)}`);
      if (!res.ok) {
        throw new Error("Failed to match inventory for location: " + incident.zone);
      }
      const matchData = await res.json();
      setInventoryMatchResult(matchData);
    } catch (err) {
      setMatchError(err.message || "Could not retrieve matching inventory.");
    } finally {
      setMatchLoading(false);
    }
  }

  // Convert Member 1's CSV field names into the incident shape used by
  // the existing Member 2 UI. Missing severity/score values stay missing.
  const integratedIncidents = (processedData?.records || []).map((record, index) => {
    const rawPriority = record.priority_level ?? record.priority ?? "Unknown";
    const priorityText = String(rawPriority).trim().toLowerCase();
    const normalizedPriority =
      priorityText === "critical" ? "Critical" :
      priorityText === "high" ? "High" :
      priorityText === "medium" ? "Medium" :
      priorityText === "low" ? "Low" : "Unknown";

    const affectedValue = Number(record.affected_population ?? record.affected ?? 0);

    return {
      ...record,
      id: record.incident_identifier ?? record.id ?? `REC-${index + 1}`,
      zone: record.incident_location ?? record.zone ?? "Unknown",
      type: record.disaster_type ?? record.type ?? "Unknown",
      affected: Number.isFinite(affectedValue) ? affectedValue : 0,
      severity: record.severity ?? null,
      priority_score: record.priority_score ?? null,
      priority: normalizedPriority,
      official_priority: normalizedPriority,
      impact_tier: record.impact_tier ?? "Unassessed",
      is_anomaly: Boolean(record.is_anomaly ?? record.anomaly),
      anomaly_status: record.anomaly_status ?? (record.is_anomaly ? "Anomaly" : "Normal"),
      anomaly_score: record.anomaly_score ?? null,
      reported_casualties: record.reported_casualties ?? null,
      injured_population: record.injured_population ?? null,
      estimated_loss: record.estimated_loss ?? null,
      contributing_evidence: record.contributing_evidence ?? ""
    };
  });

  const activeIncidents = integratedIncidents.length > 0 ? integratedIncidents : (data?.incidents || []);

  // Reconciled display data
  const displayData = data
    ? {
        ...data,
        incidents: activeIncidents,
        integratedDataset: integratedIncidents.length > 0
          ? {
              filename: processedData?.filename || "Received dataset",
              total_records: processedData?.total_records ?? integratedIncidents.length,
            }
          : null,
        recommendations: (data.recommendations && data.recommendations.length > 0)
          ? data.recommendations
          : buildIncidentRecommendations(activeIncidents, shelters, resources),
        summary: {
          ...(data.summary || {}),
          total_incidents: data.summary?.total_assessed_records ?? activeIncidents.length,
          critical_incidents: data.summary?.critical_incidents ?? 0,
          critical_impact_events: data.summary?.critical_impact_events ?? 0,
          dispatch_priority_critical_count: data.summary?.dispatch_priority_critical_count ?? 0,
          anomalies: data.summary?.anomalies_detected ?? data.summary?.anomalies ?? 0,
        },
      }
    : null;

  const currentTitle = navigation.find((item) => item[0] === page)?.[1] || "Dashboard";

  return (
    <div className="app">
      {/* SIDEBAR */}
      <aside className={`sidebar ${menuOpen ? "open" : ""}`}>
        <div className="brand">
          <div className="brand-logo">🚨</div>
          <div>
            <b>ResQ Intelligence</b>
            <small>Emergency Decision Support</small>
          </div>
          <button className="close-menu" onClick={() => setMenuOpen(false)}>✕</button>
        </div>

        <div className="workspace-label">WORKSPACE</div>

        <nav>
          {navigation.map(([id, label]) => (
            <button
              key={id}
              className={`nav-item ${page === id ? "active" : ""}`}
              onClick={() => {
                setPage(id);
                setMenuOpen(false);
              }}
            >
              <span>
                {id === "dashboard" && "▣"}
                {id === "incidents" && "🚨"}
                {id === "shelters" && "⌂"}
                {id === "resources" && "📦"}
                {id === "analytics" && "▥"}
                {id === "recommendations" && "☑"}
              </span>
              {label}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className="online-dot"></span>
            <div>
              <b>System Online</b>
              <small>FastAPI + SQLite Active</small>
            </div>
          </div>
          <div className="settings-item">
            InteroperAI Member 2
          </div>
        </div>
      </aside>

      {menuOpen && (
        <div className="sidebar-overlay" onClick={() => setMenuOpen(false)}></div>
      )}

      {/* MAIN AREA */}
      <section className="main">
        {/* TOPBAR */}
        <header className="topbar">
          <div className="header-left">
            <button className="menu-button" onClick={() => setMenuOpen(true)}>☰</button>
            <div>
              <small>DISASTER RESPONSE PLATFORM</small>
              <h1>{currentTitle}</h1>
            </div>
          </div>

          <div className="header-right">
            <span className="live-status">
              <span></span>
              Live data
            </span>
            <button className="refresh-button" onClick={loadData} title="Refresh all intelligence data">
              ↻
            </button>
            <div className="profile">M2</div>
          </div>
        </header>

        {/* CONTENT */}
        <main className="content">
          {error && <div className="error-message">⚠ {error}</div>}

          {loading && !data ? (
            <div className="loading">
              <div className="loading-icon">🚨</div>
              <b>Loading disaster intelligence...</b>
              <small>Connecting to assessment engine and persistent inventories</small>
            </div>
          ) : displayData ? (
            <>
              {page === "dashboard" && (
                <Dashboard
                  data={displayData}
                  setPage={setPage}
                  activityLogs={activityLogs}
                  shelters={shelters}
                  shelterSummary={shelterSummary}
                  resources={resources}
                  resourceSummary={resourceSummary}
                  onMatchInventory={handleOpenInventoryMatch}
                />
              )}

              {page === "incidents" && (
                <IncidentPage
                  incidents={displayData.incidents || []}
                  onMatchInventory={handleOpenInventoryMatch}
                />
              )}

              {page === "shelters" && (
                <ShelterPage
                  shelters={shelters}
                  summary={shelterSummary}
                  onRefresh={loadShelters}
                />
              )}

              {page === "resources" && (
                <ResourcePage
                  resources={resources}
                  summary={resourceSummary}
                  onRefresh={loadResources}
                />
              )}

              {page === "analytics" && (
                <AnalyticsPage
                  data={displayData}
                  shelters={shelters}
                  shelterSummary={shelterSummary}
                  resources={resources}
                  resourceSummary={resourceSummary}
                />
              )}

              {page === "recommendations" && (
                <RecommendationPage
                  recommendations={displayData.recommendations || []}
                  onMatchInventory={handleOpenInventoryMatch}
                />
              )}
            </>
          ) : null}
        </main>
      </section>

      {/* INCIDENT-TO-INVENTORY MATCHING MODAL */}
      {matchingIncident && (
        <div className="modal-backdrop" onClick={() => setMatchingIncident(null)}>
          <div className="modal-content modal-content-lg" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>📦 Incident-to-Inventory Decision Support</h3>
              <button className="modal-close" onClick={() => setMatchingIncident(null)}>✕</button>
            </div>
            <div className="modal-body">
              <div style={{ background: "#f8fafc", padding: "12px 16px", borderRadius: "8px", border: "1px solid #e2e8f0", marginBottom: "16px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
                  <b style={{ fontSize: "14px" }}>{matchingIncident.id} · {matchingIncident.type}</b>
                  <Badge text={matchingIncident.impact_tier || "Unassessed"} />
                </div>
                <div style={{ fontSize: "12px", color: "#64748b" }}>
                  <b>Location / Zone:</b> {matchingIncident.zone || "Location not provided"}
                </div>
                {matchingIncident.contributing_evidence && (
                  <div style={{ fontSize: "12px", color: "#475569", marginTop: "4px" }}>
                    <b>Evidence:</b> {matchingIncident.contributing_evidence}
                  </div>
                )}
                <div style={{ fontSize: "11px", color: "#94a3b8", marginTop: "6px" }}>
                  Match Confidence: <span style={{ color: "#3b82f6", fontWeight: "600" }}>Approximate Normalized Text Match</span> (Not Verified GPS Coordinates)
                </div>
              </div>

              {matchLoading ? (
                <div style={{ textAlign: "center", padding: "28px" }}>
                  <b>Scanning persistent facility and supply inventory...</b>
                </div>
              ) : matchError ? (
                <div className="alert-box alert-danger">⚠ {matchError}</div>
              ) : inventoryMatchResult ? (
                <>
                  {/* Warnings if full shelters exist */}
                  {inventoryMatchResult.warnings && inventoryMatchResult.warnings.length > 0 && (
                    <div style={{ marginBottom: "14px" }}>
                      {inventoryMatchResult.warnings.map((w, i) => (
                        <div key={i} className="alert-box alert-danger" style={{ fontWeight: "600" }}>
                          🚨 {w}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Matched Shelters */}
                  <h4 style={{ fontSize: "13px", fontWeight: "700", color: "#334155", marginBottom: "8px" }}>
                    Matched Shelter Facilities ({inventoryMatchResult.matched_shelters?.length || 0})
                  </h4>
                  {(!inventoryMatchResult.matched_shelters || inventoryMatchResult.matched_shelters.length === 0) ? (
                    <p style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "16px", fontStyle: "italic" }}>
                      No matching shelter facility located for '{matchingIncident.zone}'. Request local logistics field verification.
                    </p>
                  ) : (
                    <div className="table-container" style={{ marginBottom: "18px" }}>
                      <table>
                        <thead>
                          <tr>
                            <th>Shelter Name</th>
                            <th>Location</th>
                            <th>Capacity</th>
                            <th>Occupancy</th>
                            <th>Available Beds</th>
                            <th>Utilization</th>
                            <th>Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {inventoryMatchResult.matched_shelters.map((s) => (
                            <tr key={s.shelter_id} style={s.status === "Full" ? { background: "#fef2f2" } : {}}>
                              <td>
                                <b>{s.shelter_name}</b>
                                <small style={{ display: "block", color: "#94a3b8", fontSize: "10px" }}>{s.shelter_id}</small>
                              </td>
                              <td>{s.location}</td>
                              <td>{s.capacity}</td>
                              <td>{s.occupancy}</td>
                              <td><b style={{ color: s.available_beds > 0 ? "#16a34a" : "#dc2626" }}>{s.available_beds}</b></td>
                              <td>{s.utilization_percentage}%</td>
                              <td>
                                <Badge text={s.status} />
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}

                  {/* Matched Resources */}
                  <h4 style={{ fontSize: "13px", fontWeight: "700", color: "#334155", marginBottom: "8px" }}>
                    Matched Emergency Resources ({inventoryMatchResult.matched_resources?.length || 0})
                  </h4>
                  {(!inventoryMatchResult.matched_resources || inventoryMatchResult.matched_resources.length === 0) ? (
                    <p style={{ fontSize: "12px", color: "#94a3b8", fontStyle: "italic" }}>
                      No matching resource inventory found for '{matchingIncident.zone}'. Request municipal supply dispatch.
                    </p>
                  ) : (
                    <div className="table-container">
                      <table>
                        <thead>
                          <tr>
                            <th>Resource Type</th>
                            <th>Location</th>
                            <th>Required</th>
                            <th>Available</th>
                            <th>Shortage</th>
                            <th>Surplus</th>
                            <th>Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {inventoryMatchResult.matched_resources.map((r) => (
                            <tr key={r.resource_id} style={r.shortage > 0 ? { background: "#fffbeb" } : {}}>
                              <td>
                                <b>{r.resource_type}</b>
                                <small style={{ display: "block", color: "#94a3b8", fontSize: "10px" }}>{r.resource_id}</small>
                              </td>
                              <td>{r.location}</td>
                              <td>{r.required_quantity} {r.unit}</td>
                              <td>{r.available_quantity} {r.unit}</td>
                              <td><b style={{ color: r.shortage > 0 ? "#dc2626" : "#475569" }}>{r.shortage} {r.unit}</b></td>
                              <td><b style={{ color: r.surplus > 0 ? "#16a34a" : "#475569" }}>{r.surplus} {r.unit}</b></td>
                              <td><Badge text={r.status} /></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}

                  <div style={{ marginTop: "14px", padding: "10px 14px", background: "#f8fafc", borderRadius: "6px", fontSize: "11px", color: "#64748b", border: "1px solid #e2e8f0" }}>
                    ℹ️ <b>Operational Disclaimer:</b> Decision support matching is based on approximate normalized text. This tool does not execute automatic 911 dispatch or resource deployments without authorized human review.
                  </div>
                </>
              ) : null}
            </div>
            <div className="modal-footer">
              <button className="btn-outline" onClick={() => setMatchingIncident(null)}>Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ============================================================
   DATA UTILITIES & DYNAMIC LOSS COVERAGE
   ============================================================ */

export function isValidLoss(val) {
  if (val === null || val === undefined || val === "") return false;
  const n = Number(val);
  return !isNaN(n) && isFinite(n);
}

export function computeLossCoverage(incidents = []) {
  const total = incidents.length;
  if (total === 0) {
    return {
      total: 0,
      assessed: 0,
      unassessed: 0,
      coveragePct: "0.0",
      unassessedPct: "0.0",
      display: "0.0% Present (0.0% Unassessed)"
    };
  }
  const assessed = incidents.filter((i) => isValidLoss(i.estimated_loss)).length;
  const unassessed = total - assessed;
  const coveragePct = ((assessed / total) * 100).toFixed(1);
  const unassessedPct = ((unassessed / total) * 100).toFixed(1);
  return {
    total,
    assessed,
    unassessed,
    coveragePct,
    unassessedPct,
    display: `${coveragePct}% Present (${unassessedPct}% Unassessed)`
  };
}

/* ============================================================
   DASHBOARD
   ============================================================ */

function Dashboard({
  data,
  setPage,
  activityLogs = [],
  shelters = [],
  shelterSummary = null,
  resources = [],
  resourceSummary = null,
  onMatchInventory
}) {
  const summary = data.summary || {};
  const incidents = data.incidents || [];

  // Dynamic economic loss calculation across active assessment records
  const totalRecords = incidents.length;
  const lossStats = computeLossCoverage(incidents);
  const economicLossCoverageText =
    summary?.data_quality_report?.economic_loss_coverage?.coverage_display ||
    lossStats.display;

  // Dynamic casualty & loss totals
  const totalReportedCasualties = incidents.reduce((acc, i) => acc + (Number(i.reported_casualties) || 0), 0);
  const totalReportedInjuries = incidents.reduce((acc, i) => acc + (Number(i.injured_population) || 0), 0);
  const totalEstimatedLoss = incidents.reduce((acc, i) => acc + (Number(i.estimated_loss) || 0), 0);

  // Active inventory metrics
  const totalShelters = shelterSummary?.total_shelters ?? shelters.length;
  const totalCapacity = shelterSummary?.total_capacity ?? shelters.reduce((a, s) => a + (s.capacity || 0), 0);
  const totalOccupancy = shelterSummary?.total_occupancy ?? shelters.reduce((a, s) => a + (s.occupancy || 0), 0);
  const totalAvailableBeds = shelterSummary?.total_available_beds ?? (totalCapacity - totalOccupancy);
  const nearCapacityShelters = shelterSummary?.near_capacity_count ?? shelters.filter((s) => s.status === "Near Capacity").length;
  const fullShelters = shelterSummary?.full_count ?? shelters.filter((s) => s.status === "Full").length;

  const totalResourceTypes = resourceSummary?.resource_types_tracked ?? new Set(resources.map((r) => r.resource_type)).size;
  const shortageCount = resourceSummary?.shortage_count ?? resources.filter((r) => (r.shortage || 0) > 0).length;
  const largestShortages = resourceSummary?.largest_shortages ?? resources.filter((r) => (r.shortage || 0) > 0).slice(0, 3);

  // Inventory source labels
  const isDemoShelter = shelters.some((s) => s.data_source && s.data_source.includes("DEMO"));
  const isDemoResource = resources.some((r) => r.data_source && r.data_source.includes("DEMO"));

  return (
    <>
      {/* WELCOME SECTION */}
      <section className="welcome-section">
        <div>
          <div className="eyebrow">INTELLIGENT DISASTER RESPONSE · MEMBER 2</div>
          <h2>Multi-Hazard Operational Intelligence</h2>
          <p>
            Real-time evidence-based assessment combining NOAA Storm Events telemetry, Isolation Forest anomaly detection, and persistent municipal relief inventory.
          </p>
        </div>

        <div className="risk-badge-container">
          <div className="risk-badge">
            <small>SCREENED SEVERITY PROFILE</small>
            <b>{summary.overall_risk || "Insufficient evidence"}</b>
            {summary.overall_risk_reason && (
              <small className="risk-reason-text">
                {summary.overall_risk_reason}
              </small>
            )}
            <small className="risk-disclaimer-text">
              * Rule-based screening from observed casualties & property losses · Not an automated dispatch declaration
            </small>
          </div>
        </div>
      </section>

      {/* STATISTICS */}
      <section className="stats-grid">
        <StatCard
          label="Assessed Records"
          value={summary.total_assessed_records ?? totalRecords}
          subtext="Active Ingestion Assessment"
          icon="📊"
        />

        <StatCard
          label="Critical Impact Events"
          value={summary.critical_impact_events ?? 0}
          danger={Number(summary.critical_impact_events) > 0}
          subtext="Fatalities > 0 or Loss ≥ $5M"
          icon="🚨"
        />

        <StatCard
          label="Official Dispatch Priority"
          value={
            summary.dispatch_priority_critical_count !== undefined && summary.dispatch_priority_critical_count !== null
              ? `${summary.dispatch_priority_critical_count} Critical`
              : "Unavailable"
          }
          subtext="100% Unknown in NOAA Telemetry"
          icon="🏷️"
        />

        <StatCard
          label="Unusual Records"
          value={summary.anomalies_detected ?? summary.anomalies ?? 0}
          subtext="Isolation Forest Outliers"
          icon="🔍"
        />

        <StatCard
          label="Available Shelter Beds"
          value={totalShelters > 0 ? `${totalAvailableBeds.toLocaleString()} Beds` : "No Shelters"}
          subtext={totalShelters > 0 ? `${nearCapacityShelters} Near Cap · ${fullShelters} Full` : "Add or Reset Demo"}
          warning={fullShelters > 0}
          icon="⌂"
        />

        <StatCard
          label="Supply Shortages"
          value={resources.length > 0 ? `${shortageCount} Shortages` : "No Resources"}
          subtext={resources.length > 0 ? `${totalResourceTypes} Types Tracked` : "Add or Reset Demo"}
          danger={shortageCount > 0}
          icon="📦"
        />
      </section>

      {/* TWO COLUMN DASHBOARD */}
      <section className="dashboard-grid">
        {/* INCIDENTS PANEL */}
        <Panel
          title="Critical & High Impact Incidents"
          action="View all"
          onClick={() => setPage("incidents")}
        >
          {incidents.slice(0, 5).map((incident) => (
            <div className="incident-row" key={incident.id}>
              <div className="incident-details">
                <b style={{ display: "block", color: "#0f172a", fontSize: "12px" }}>{incident.id}</b>
                <small style={{ color: "#64748b" }}>
                  {incident.zone} · {incident.type}
                </small>
                {incident.contributing_evidence && (
                  <small style={{ display: "block", color: "#475569", marginTop: "2px" }}>
                    {incident.contributing_evidence}
                  </small>
                )}
              </div>
              <div className="incident-actions">
                <Badge text={incident.impact_tier || "Unassessed"} />
                <button
                  className="match-action-btn"
                  onClick={() => onMatchInventory(incident)}
                  title="Inspect matching shelter and supply inventory"
                >
                  📦 Match
                </button>
              </div>
            </div>
          ))}
          {incidents.length === 0 && (
            <div className="analytics-item">
              <b>No incident records loaded</b>
              <small>Upload a disaster dataset via Member 1 receiver.</small>
            </div>
          )}
        </Panel>

        {/* SHELTERS PANEL */}
        <Panel
          title="Shelter Capacity & Occupancy"
          action="Manage Shelters"
          onClick={() => setPage("shelters")}
        >
          <div className="panel-sub-header">
            <span style={{ fontSize: "11px", color: "#64748b" }}>
              Total Shelters: <b>{totalShelters}</b> · Available: <b>{totalAvailableBeds.toLocaleString()} beds</b>
            </span>
            <span className={`badge ${isDemoShelter ? "demo-source" : "user-source"}`}>
              {isDemoShelter ? "DEMO INVENTORY" : "USER INVENTORY"}
            </span>
          </div>

          {shelters.length === 0 ? (
            <div className="analytics-item" style={{ padding: "16px" }}>
              <b>No Shelter Inventory Connected</b>
              <small style={{ display: "block", marginTop: "4px" }}>
                Navigate to Shelter Management to add records or reset to verified demo facilities.
              </small>
            </div>
          ) : (
            shelters.slice(0, 4).map((shelter) => (
              <div className="shelter-row" key={shelter.shelter_id}>
                <div className="row-header">
                  <div className="shelter-info">
                    <b>{shelter.shelter_name}</b>
                    <small>{shelter.location} · {shelter.shelter_id}</small>
                  </div>
                  <Badge text={shelter.status} />
                </div>
                <div className="progress-bar">
                  <span
                    style={{
                      width: `${Math.min(shelter.utilization_percentage || 0, 100)}%`,
                      background: shelter.status === "Full" ? "#ef4444" : shelter.status === "Near Capacity" ? "#f59e0b" : "#22c55e"
                    }}
                  ></span>
                </div>
                <div className="progress-info">
                  <span>{shelter.occupancy} / {shelter.capacity} occupied</span>
                  <b>{shelter.available_beds} beds available ({shelter.utilization_percentage}%)</b>
                </div>
              </div>
            ))
          )}
        </Panel>

        {/* RESOURCES PANEL */}
        <Panel
          title="Emergency Resources & Shortages"
          action="Manage Resources"
          onClick={() => setPage("resources")}
        >
          <div className="panel-sub-header">
            <span style={{ fontSize: "11px", color: "#64748b" }}>
              Shortages Detected: <b style={{ color: shortageCount > 0 ? "#dc2626" : "#16a34a" }}>{shortageCount}</b>
            </span>
            <span className={`badge ${isDemoResource ? "demo-source" : "user-source"}`}>
              {isDemoResource ? "DEMO INVENTORY" : "USER INVENTORY"}
            </span>
          </div>

          {resources.length === 0 ? (
            <div className="analytics-item" style={{ padding: "16px" }}>
              <b>No Resource Inventory Connected</b>
              <small style={{ display: "block", marginTop: "4px" }}>
                Navigate to Resource Management to track water, medical kits, and relief supplies.
              </small>
            </div>
          ) : (
            resources.slice(0, 4).map((resource) => (
              <div className="resource-row" key={resource.resource_id}>
                <div className="warning-icon" style={{ color: resource.shortage > 0 ? "#ef4444" : "#22c55e" }}>
                  {resource.shortage > 0 ? "⚠" : "✓"}
                </div>
                <div className="resource-info">
                  <b>{resource.resource_type}</b>
                  <small>
                    {resource.location} · Required: {resource.required_quantity} {resource.unit} | Stock: {resource.available_quantity} {resource.unit}
                    {resource.shortage > 0 && <span style={{ color: "#dc2626", fontWeight: "700" }}> (Deficit: {resource.shortage})</span>}
                  </small>
                </div>
                <Badge text={resource.status} />
              </div>
            ))
          )}
        </Panel>

        {/* ANOMALIES PANEL */}
        <Panel
          title="Anomaly Detection (Isolation Forest)"
          action="Analytics"
          onClick={() => setPage("analytics")}
        >
          <div className="anomaly-context-banner">
            <small>
              <b>Statistical Outlier Review:</b> Flagged records represent multidimensional outliers learned by Isolation Forest. An unusual record represents atypical incident telemetry (e.g. high property loss or rare coordinates) requiring human verification.
            </small>
          </div>

          {(data.anomalies || []).length === 0 ? (
            <div className="analytics-item">
              <b>No anomaly data available</b>
              <small>Isolation Forest has not returned flagged records.</small>
            </div>
          ) : (
            (data.anomalies || []).slice(0, 4).map((item, index) => (
              <div className="anomaly-row" key={index}>
                <span className="anomaly-icon" style={{ color: item.anomaly ? "#ef4444" : "#22c55e" }}>
                  {item.anomaly ? "⚠" : "✓"}
                </span>
                <div className="anomaly-content">
                  <div className="anomaly-header">
                    <b>{item.id ? `${item.id} · ${item.type || 'Incident'}` : item.shelter}</b>
                    {item.anomaly_score != null && (
                      <span className="anomaly-score-tag">
                        Score: {Number(item.anomaly_score).toFixed(4)}
                      </span>
                    )}
                  </div>
                  <small style={{ display: "block", color: "var(--text-muted, #71839b)", marginTop: "2px" }}>
                    {item.zone || "Location not specified"}
                    {item.contributing_evidence ? ` · ${item.contributing_evidence}` : ""}
                  </small>
                </div>
                <Badge text={item.status || (item.anomaly ? "Unusual" : "Normal")} />
              </div>
            ))
          )}
        </Panel>
      </section>

      {/* PROVENANCE AND DATA QUALITY AUDIT */}
      <Panel title="Assessment Provenance & Data Quality Traceability">
        <div className="provenance-grid">
          <div className="provenance-card">
            <div className="provenance-card-title">ACTIVE DATASET INGESTION</div>
            <div className="provenance-item">
              <span>Source File:</span>
              <code>{data.integratedDataset?.filename || summary.filename || "NOAA Telemetry"}</code>
            </div>
            <div className="provenance-item">
              <span>Records Evaluated:</span>
              <b>{summary.total_assessed_records ?? totalRecords}</b>
            </div>
            <div className="provenance-item">
              <span>Source System:</span>
              <span>Member 1 Pipeline</span>
            </div>
            <div className="provenance-item">
              <span>Assessment ID:</span>
              <code>{summary.assessment_id || "Local Assessment"}</code>
            </div>
            <div className="provenance-item">
              <span>Job ID:</span>
              <code>{summary.job_id || "Active Job"}</code>
            </div>
          </div>

          <div className="provenance-card">
            <div className="provenance-card-title">EVIDENCE COMPLETENESS & GAPS</div>
            <div className="provenance-item">
              <span>Casualties & Injuries:</span>
              <span style={{ color: "#22c55e", fontWeight: "600" }}>Verified Present</span>
            </div>
            <div className="provenance-item">
              <span>Economic Loss Coverage:</span>
              <span style={{ color: "#0f172a", fontWeight: "700" }}>{economicLossCoverageText}</span>
            </div>
            <div className="provenance-item">
              <span>Official CAD Priority:</span>
              <span style={{ color: "#f59e0b", fontWeight: "600" }}>100% Unknown (CAD Disconnected)</span>
            </div>
            <div className="provenance-item">
              <span>Inventory Storage:</span>
              <span style={{ color: "#2563eb", fontWeight: "600" }}>Persistent SQLite Database</span>
            </div>
          </div>

          <div className="provenance-card">
            <div className="provenance-card-title">OPERATIONAL AUDIT TRAIL</div>
            <div className="provenance-item">
              <span>Total Deaths / Injuries:</span>
              <b>{totalReportedCasualties} / {totalReportedInjuries}</b>
            </div>
            <div className="provenance-item">
              <span>Total Estimated Damage:</span>
              <b>${totalEstimatedLoss.toLocaleString()}</b>
            </div>
            <div className="provenance-item">
              <span>Anomalies Flagged:</span>
              <b>{summary.anomalies_detected ?? 0} ({totalRecords > 0 ? (((summary.anomalies_detected ?? 0) / totalRecords) * 100).toFixed(1) : 0}%)</b>
            </div>
            <div className="provenance-item">
              <span>Inventory Match Mode:</span>
              <span>Approximate Text Heuristic</span>
            </div>
          </div>
        </div>

        {/* RECENT ACTIVITY LOGS */}
        <div className="analytics-list">
          <div style={{ padding: "8px 0 4px", fontSize: "11px", fontWeight: "700", color: "#64748b", letterSpacing: "0.5px" }}>
            RECENT INGESTION TRANSACTIONS
          </div>
          {activityLogs && activityLogs.length > 0 ? (
            activityLogs.slice(0, 3).map((log, index) => (
              <div className="analytics-item" key={index} style={{ padding: "10px 14px" }}>
                <div className="row-header" style={{ marginBottom: "4px" }}>
                  <div>
                    <b>{log.filename || "Dataset Ingestion"}</b>
                    <small style={{ color: "#64748b" }}>
                      Rows: {log.rows_received ?? "N/A"} · Size: {log.file_size_bytes ? `${Math.round(log.file_size_bytes / 1024)} KB` : "N/A"}
                    </small>
                  </div>
                  <Badge text={log.status || "Completed"} />
                </div>
                <div style={{ fontSize: "11px", color: "#64748b", display: "flex", gap: "12px", flexWrap: "wrap" }}>
                  <span>Job: <code>{log.job_id || "N/A"}</code></span>
                  <span>Time: {log.timestamp ? new Date(log.timestamp).toLocaleString() : "Recent"}</span>
                </div>
              </div>
            ))
          ) : (
            <div className="analytics-item">
              <b>No API upload activity recorded in current session</b>
              <small>Upload records via <code>POST /api/receive-file</code> from Member 1 to trace requests.</small>
            </div>
          )}
        </div>
      </Panel>

      <Footer />
    </>
  );
}

/* ============================================================
   INCIDENT PAGE
   ============================================================ */

function IncidentPage({ incidents, onMatchInventory }) {
  const [filter, setFilter] = useState("All");
  const [search, setSearch] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 25;

  function matchFilter(incident, filterName) {
    if (filterName === "All") return true;
    const tier = incident.impact_tier || "";
    if (filterName === "Critical") return tier.includes("Critical");
    if (filterName === "High") return tier.includes("High");
    if (filterName === "Medium") return tier.includes("Moderate") || tier.includes("Medium");
    if (filterName === "Low") return tier.includes("Minor") || tier.includes("Negligible") || tier.includes("Low");
    if (filterName === "Unknown") {
      return (
        tier.includes("Unassessed") ||
        tier.includes("Indeterminate") ||
        tier.includes("Unknown") ||
        (!tier && (incident.official_priority === "Unknown" || incident.priority === "Unknown"))
      );
    }
    if (filterName === "Anomalies") {
      return Boolean(incident.is_anomaly || incident.anomaly_status === "Anomaly" || incident.anomaly_status === "Unusual" || incident.status === "Unusual");
    }
    return true;
  }

  const filtered = incidents.filter((incident) => {
    // Search match
    if (search.trim()) {
      const q = search.toLowerCase();
      const idMatch = String(incident.id || "").toLowerCase().includes(q);
      const locMatch = String(incident.zone || incident.incident_location || "").toLowerCase().includes(q);
      const typeMatch = String(incident.type || incident.disaster_type || "").toLowerCase().includes(q);
      if (!idMatch && !locMatch && !typeMatch) return false;
    }

    // Filter match based on rule-based impact tier or anomaly status
    return matchFilter(incident, filter);
  });

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const safePage = Math.min(currentPage, totalPages);
  const visibleIncidents = filtered.slice((safePage - 1) * pageSize, safePage * pageSize);

  return (
    <>
      <PageIntro
        tag="ASSESSMENT ENGINE"
        title="Incident Priority & Impact Assessment"
        text="Browse integrated disaster records received from Member 1. Rule-based impact tiers are derived from reported casualties and economic damage. Official municipal dispatch priority remains Unknown for NOAA telemetry."
      />

      <div className="toolbar-container">
        <div className="filters" style={{ margin: 0 }}>
          {["All", "Critical", "High", "Medium", "Low", "Unknown", "Anomalies"].map((filterName) => {
            const count = incidents.filter((inc) => matchFilter(inc, filterName)).length;
            return (
              <button
                key={filterName}
                className={filter === filterName ? "filter active" : "filter"}
                onClick={() => {
                  setFilter(filterName);
                  setCurrentPage(1);
                }}
              >
                {filterName}
                <span style={{ marginLeft: "5px", fontSize: "10px", opacity: 0.85 }}>
                  ({count})
                </span>
              </button>
            );
          })}
        </div>

        <input
          type="text"
          className="search-input"
          placeholder="Search incident, location, or type..."
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setCurrentPage(1);
          }}
        />
      </div>

      <Panel>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Incident ID & Type</th>
                <th>Location / Zone</th>
                <th>Rule-Based Impact Tier</th>
                <th>Casualties & Damage Evidence</th>
                <th>Official CAD Priority</th>
                <th>Anomaly Status</th>
                <th>Inventory Action</th>
              </tr>
            </thead>
            <tbody>
              {visibleIncidents.map((incident) => (
                <tr key={incident.id}>
                  <td>
                    <b>{incident.id}</b>
                    <small style={{ display: "block", color: "#64748b" }}>{incident.type}</small>
                  </td>
                  <td>{incident.zone}</td>
                  <td>
                    <Badge text={incident.impact_tier || "Unassessed"} />
                  </td>
                  <td>
                    <div style={{ fontSize: "11px", color: "#334155" }}>
                      <span>Deaths: <b>{incident.reported_casualties ?? 0}</b></span> ·{" "}
                      <span>Injured: <b>{incident.injured_population ?? 0}</b></span> ·{" "}
                      <span>Loss: <b>${incident.estimated_loss != null ? Number(incident.estimated_loss).toLocaleString() : "Unassessed"}</b></span>
                    </div>
                  </td>
                  <td>
                    <span className="badge badge-neutral">
                      {incident.official_priority || "Unknown"}
                    </span>
                  </td>
                  <td>
                    {incident.is_anomaly || incident.anomaly_status === "Anomaly" ? (
                      <span className="badge badge-unusual">
                        ⚠ Outlier ({Number(incident.anomaly_score || 0).toFixed(3)})
                      </span>
                    ) : (
                      <span className="badge badge-normal">✓ Normal</span>
                    )}
                  </td>
                  <td>
                    <button
                      className="match-action-btn"
                      onClick={() => onMatchInventory(incident)}
                    >
                      📦 Match Inventory
                    </button>
                  </td>
                </tr>
              ))}

              {visibleIncidents.length === 0 && (
                <tr>
                  <td colSpan={7} style={{ textAlign: "center", padding: "28px" }}>
                    No incident records match the active '{filter}' filter.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="pagination-controls" style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "12px",
          padding: "14px 4px 2px",
          flexWrap: "wrap"
        }}>
          <small>
            Showing {filtered.length === 0 ? 0 : (safePage - 1) * pageSize + 1}
            –{Math.min(safePage * pageSize, filtered.length)} of {filtered.length.toLocaleString()} records
          </small>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <button
              className="btn-outline btn-sm"
              disabled={safePage <= 1}
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            >
              Previous
            </button>
            <span style={{ fontSize: "12px", color: "#64748b" }}>
              Page {safePage} of {totalPages}
            </span>
            <button
              className="btn-outline btn-sm"
              disabled={safePage >= totalPages}
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            >
              Next
            </button>
          </div>
        </div>
      </Panel>

      <Footer />
    </>
  );
}

/* ============================================================
   SHELTER MANAGEMENT
   ============================================================ */

function ShelterPage({ shelters = [], summary = null, onRefresh }) {
  const [filter, setFilter] = useState("All");
  const [search, setSearch] = useState("");
  const [modalOpen, setModalOpen] = useState(false);
  const [editingShelter, setEditingShelter] = useState(null);
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [resetConfirmOpen, setResetConfirmOpen] = useState(false);
  const [deleteId, setDeleteId] = useState(null);

  const [formData, setFormData] = useState({
    shelter_id: "",
    shelter_name: "",
    location: "",
    capacity: "",
    occupancy: "",
    notes: ""
  });
  const [formError, setFormError] = useState("");
  const [actionSuccess, setActionSuccess] = useState("");

  const filteredShelters = shelters.filter((s) => {
    if (search.trim()) {
      const q = search.toLowerCase();
      const nMatch = (s.shelter_name || "").toLowerCase().includes(q);
      const lMatch = (s.location || "").toLowerCase().includes(q);
      const idMatch = (s.shelter_id || "").toLowerCase().includes(q);
      if (!nMatch && !lMatch && !idMatch) return false;
    }
    if (filter === "All") return true;
    return s.status === filter;
  });

  function openCreateModal() {
    setEditingShelter(null);
    setFormData({
      shelter_id: "",
      shelter_name: "",
      location: "",
      capacity: "",
      occupancy: "",
      notes: ""
    });
    setFormError("");
    setModalOpen(true);
  }

  function openEditModal(shelter) {
    setEditingShelter(shelter);
    setFormData({
      shelter_id: shelter.shelter_id,
      shelter_name: shelter.shelter_name,
      location: shelter.location,
      capacity: shelter.capacity,
      occupancy: shelter.occupancy,
      notes: shelter.notes || ""
    });
    setFormError("");
    setModalOpen(true);
  }

  async function handleSaveShelter(e) {
    e.preventDefault();
    setFormError("");

    const cap = parseInt(formData.capacity, 10);
    const occ = parseInt(formData.occupancy, 10);

    if (!formData.shelter_name.trim() || !formData.location.trim()) {
      setFormError("Shelter name and location cannot be empty.");
      return;
    }
    if (isNaN(cap) || cap <= 0) {
      setFormError("Capacity must be a positive integer.");
      return;
    }
    if (isNaN(occ) || occ < 0) {
      setFormError("Occupancy must be a non-negative integer.");
      return;
    }
    if (occ > cap) {
      setFormError(`Occupancy (${occ}) cannot exceed capacity (${cap}).`);
      return;
    }

    try {
      let res;
      if (editingShelter) {
        res = await fetch(`${API}/api/shelters/${encodeURIComponent(editingShelter.shelter_id)}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(formData)
        });
      } else {
        res = await fetch(`${API}/api/shelters`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(formData)
        });
      }

      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || "Failed to save shelter record.");
      }

      setModalOpen(false);
      setActionSuccess(editingShelter ? "Shelter updated successfully!" : "Shelter created successfully!");
      setTimeout(() => setActionSuccess(""), 4000);
      onRefresh();
    } catch (err) {
      setFormError(err.message);
    }
  }

  async function handleDeleteShelter() {
    if (!deleteId) return;
    try {
      const res = await fetch(`${API}/api/shelters/${encodeURIComponent(deleteId)}`, {
        method: "DELETE"
      });
      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || "Failed to delete shelter.");
      }
      setDeleteId(null);
      setActionSuccess("Shelter deleted successfully.");
      setTimeout(() => setActionSuccess(""), 4000);
      onRefresh();
    } catch (err) {
      alert("Error deleting shelter: " + err.message);
    }
  }

  async function handleResetToDemo() {
    try {
      const res = await fetch(`${API}/api/shelters/reset`, { method: "POST" });
      if (!res.ok) throw new Error("Failed to reset shelters.");
      setResetConfirmOpen(false);
      setActionSuccess("Shelters successfully reset to verified demo dataset.");
      setTimeout(() => setActionSuccess(""), 4000);
      onRefresh();
    } catch (err) {
      alert(err.message);
    }
  }

  async function handleImportCSV(e) {
    const file = e.target.files[0];
    if (!file) return;

    const body = new FormData();
    body.append("file", file);

    try {
      const res = await fetch(`${API}/api/shelters/import`, {
        method: "POST",
        body
      });
      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || "Import failed.");
      }
      const data = await res.json();
      setImportModalOpen(false);
      setActionSuccess(`Successfully imported ${data.imported_count} shelter records!`);
      setTimeout(() => setActionSuccess(""), 4000);
      onRefresh();
    } catch (err) {
      alert("Import error: " + err.message);
    }
  }

  return (
    <>
      <PageIntro
        tag="CAPACITY MONITORING"
        title="Shelter Management & Inventory"
        text="Manage persistent shelter facilities across operational zones. Track certified bed capacities, intake headcounts, available beds, and utilization thresholds in SQLite."
      />

      {/* ACTION MESSAGES */}
      {actionSuccess && (
        <div className="alert-box alert-success" style={{ marginBottom: "16px" }}>
          ✓ {actionSuccess}
        </div>
      )}

      {/* METRIC CARDS */}
      <div className="stats-grid" style={{ marginBottom: "20px" }}>
        <StatCard
          label="Total Facilities"
          value={summary?.total_shelters ?? shelters.length}
          subtext="Persistent SQLite Records"
          icon="⌂"
        />
        <StatCard
          label="Total Bed Capacity"
          value={(summary?.total_capacity ?? 0).toLocaleString()}
          subtext="Maximum Certified Intake"
          icon="🛏️"
        />
        <StatCard
          label="Current Occupancy"
          value={(summary?.total_occupancy ?? 0).toLocaleString()}
          subtext={`Overall: ${summary?.overall_utilization_pct ?? 0}%`}
          icon="👥"
        />
        <StatCard
          label="Available Beds"
          value={(summary?.total_available_beds ?? 0).toLocaleString()}
          subtext="Unoccupied Bed Capacity"
          icon="✓"
        />
        <StatCard
          label="Near Capacity"
          value={summary?.near_capacity_count ?? 0}
          warning={(summary?.near_capacity_count ?? 0) > 0}
          subtext="80% – 99% Utilized"
          icon="⚠"
        />
        <StatCard
          label="Full Facilities"
          value={summary?.full_count ?? 0}
          danger={(summary?.full_count ?? 0) > 0}
          subtext="100% Capacity (Intake Blocked)"
          icon="🚨"
        />
      </div>

      {/* TOOLBAR */}
      <div className="toolbar-container">
        <div className="filters" style={{ margin: 0 }}>
          {["All", "Available", "Near Capacity", "Full"].map((st) => (
            <button
              key={st}
              className={filter === st ? "filter active" : "filter"}
              onClick={() => setFilter(st)}
            >
              {st}
            </button>
          ))}
        </div>

        <div className="toolbar-actions">
          <input
            type="text"
            className="search-input"
            placeholder="Search shelter or location..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <button className="btn-primary" onClick={openCreateModal}>
            + Add Shelter
          </button>
          <a
            href={`${API}/api/shelters/template`}
            download="shelters_template.csv"
            className="btn-outline"
            style={{ textDecoration: "none" }}
          >
            📥 CSV Template
          </a>
          <button className="btn-outline" onClick={() => setImportModalOpen(true)}>
            📤 Import CSV
          </button>
          <button className="btn-outline" onClick={() => setResetConfirmOpen(true)}>
            ↻ Reset to Demo
          </button>
        </div>
      </div>

      {/* SHELTER TABLE */}
      <Panel>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Shelter ID & Name</th>
                <th>Location / Zone</th>
                <th>Capacity</th>
                <th>Occupancy</th>
                <th>Available Beds</th>
                <th>Utilization</th>
                <th>Status</th>
                <th>Data Source</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredShelters.map((s) => (
                <tr key={s.shelter_id} style={s.status === "Full" ? { background: "#fef2f2" } : {}}>
                  <td>
                    <b>{s.shelter_name}</b>
                    <small style={{ display: "block", color: "#64748b" }}>{s.shelter_id}</small>
                    {s.notes && <small style={{ display: "block", color: "#94a3b8", fontStyle: "italic" }}>{s.notes}</small>}
                  </td>
                  <td>{s.location}</td>
                  <td><b>{s.capacity.toLocaleString()}</b></td>
                  <td>{s.occupancy.toLocaleString()}</td>
                  <td>
                    <b style={{ color: s.available_beds > 0 ? "#16a34a" : "#dc2626" }}>
                      {s.available_beds.toLocaleString()}
                    </b>
                  </td>
                  <td>
                    <div>{s.utilization_percentage}%</div>
                    <div className="progress-bar" style={{ height: "5px", margin: "4px 0 0" }}>
                      <span
                        style={{
                          width: `${Math.min(s.utilization_percentage || 0, 100)}%`,
                          background: s.status === "Full" ? "#ef4444" : s.status === "Near Capacity" ? "#f59e0b" : "#22c55e"
                        }}
                      ></span>
                    </div>
                  </td>
                  <td>
                    <Badge text={s.status} />
                  </td>
                  <td>
                    <span className={`badge ${s.data_source && s.data_source.includes("DEMO") ? "demo-source" : "user-source"}`}>
                      {s.data_source && s.data_source.includes("DEMO") ? "DEMO DATA" : "USER SUPPLIED"}
                    </span>
                  </td>
                  <td>
                    <div style={{ display: "flex", gap: "6px" }}>
                      <button className="btn-outline btn-sm" onClick={() => openEditModal(s)}>Edit</button>
                      <button className="btn-danger btn-sm" onClick={() => setDeleteId(s.shelter_id)}>Delete</button>
                    </div>
                  </td>
                </tr>
              ))}

              {filteredShelters.length === 0 && (
                <tr>
                  <td colSpan={9} style={{ textAlign: "center", padding: "32px" }}>
                    No shelter facilities match the current criteria.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Panel>

      {/* CREATE / EDIT MODAL */}
      {modalOpen && (
        <div className="modal-backdrop" onClick={() => setModalOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{editingShelter ? `Edit Shelter (${editingShelter.shelter_id})` : "Add New Shelter Facility"}</h3>
              <button className="modal-close" onClick={() => setModalOpen(false)}>✕</button>
            </div>
            <form onSubmit={handleSaveShelter}>
              <div className="modal-body">
                {formError && <div className="alert-box alert-danger">{formError}</div>}

                <div className="form-group">
                  <label>Shelter ID {editingShelter ? "*" : "(Optional — auto-generated if left blank)"}</label>
                  <input
                    type="text"
                    className="form-input"
                    value={formData.shelter_id}
                    onChange={(e) => setFormData({ ...formData, shelter_id: e.target.value })}
                    placeholder="e.g. SH-06 (or leave blank to auto-generate)"
                    disabled={!!editingShelter}
                  />
                  <small className="form-hint">
                    {editingShelter
                      ? "Shelter ID cannot be modified."
                      : "Optional unique identifier. A standard sequential ID (e.g. SH-06) will be assigned if left blank."}
                  </small>
                </div>

                <div className="form-group">
                  <label>Shelter Name *</label>
                  <input
                    type="text"
                    className="form-input"
                    value={formData.shelter_name}
                    onChange={(e) => setFormData({ ...formData, shelter_name: e.target.value })}
                    placeholder="e.g. Metro Civic Evacuation Center"
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Location / Zone *</label>
                  <input
                    type="text"
                    className="form-input"
                    value={formData.location}
                    onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                    placeholder="e.g. San Juan, Metro Valley, Okanogan"
                    required
                  />
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
                  <div className="form-group">
                    <label>Capacity (Beds) *</label>
                    <input
                      type="number"
                      min="1"
                      className="form-input"
                      value={formData.capacity}
                      onChange={(e) => setFormData({ ...formData, capacity: e.target.value })}
                      placeholder="e.g. 500"
                      required
                    />
                    <small className="form-hint">Must be greater than 0</small>
                  </div>

                  <div className="form-group">
                    <label>Current Occupancy *</label>
                    <input
                      type="number"
                      min="0"
                      className="form-input"
                      value={formData.occupancy}
                      onChange={(e) => setFormData({ ...formData, occupancy: e.target.value })}
                      placeholder="e.g. 250"
                      required
                    />
                    <small className="form-hint">Must not exceed capacity</small>
                  </div>
                </div>

                <div className="form-group">
                  <label>Notes & Resources (Optional)</label>
                  <textarea
                    className="form-textarea"
                    value={formData.notes}
                    onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
                    placeholder="Emergency generator on-site, ADA accessible, kitchen depot..."
                  />
                </div>
              </div>
              <div className="modal-footer">
                <button type="button" className="btn-outline" onClick={() => setModalOpen(false)}>Cancel</button>
                <button type="submit" className="btn-primary">{editingShelter ? "Save Changes" : "Create Shelter"}</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* CSV IMPORT MODAL */}
      {importModalOpen && (
        <div className="modal-backdrop" onClick={() => setImportModalOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Bulk Import Shelters via CSV</h3>
              <button className="modal-close" onClick={() => setImportModalOpen(false)}>✕</button>
            </div>
            <div className="modal-body">
              <p style={{ fontSize: "13px", color: "#475569", marginBottom: "14px" }}>
                Upload a structured CSV file with columns: <code>shelter_name</code>, <code>location</code>, <code>capacity</code>, <code>occupancy</code>.
              </p>
              <div className="form-group">
                <input
                  type="file"
                  accept=".csv"
                  className="form-input"
                  onChange={handleImportCSV}
                />
              </div>
              <small className="form-hint">
                Need the standard template? <a href={`${API}/api/shelters/template`} download>Download shelters_template.csv</a>
              </small>
            </div>
            <div className="modal-footer">
              <button className="btn-outline" onClick={() => setImportModalOpen(false)}>Cancel</button>
            </div>
          </div>
        </div>
      )}

      {/* RESET CONFIRMATION */}
      {resetConfirmOpen && (
        <div className="modal-backdrop" onClick={() => setResetConfirmOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Reset Shelter Inventory to Demo?</h3>
              <button className="modal-close" onClick={() => setResetConfirmOpen(false)}>✕</button>
            </div>
            <div className="modal-body">
              <p style={{ fontSize: "13px", color: "#475569" }}>
                This action will restore the verified demo shelter dataset (including near-capacity and 100% full facilities) and overwrite current custom edits.
              </p>
              <div className="alert-box alert-warning" style={{ marginTop: "12px" }}>
                All seeded records are explicitly marked: <b>DEMO DATA — NOT LIVE MUNICIPAL INVENTORY</b>.
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn-outline" onClick={() => setResetConfirmOpen(false)}>Cancel</button>
              <button className="btn-danger" onClick={handleResetToDemo}>Confirm Reset</button>
            </div>
          </div>
        </div>
      )}

      {/* DELETE CONFIRMATION */}
      {deleteId && (
        <div className="modal-backdrop" onClick={() => setDeleteId(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Confirm Shelter Deletion</h3>
              <button className="modal-close" onClick={() => setDeleteId(null)}>✕</button>
            </div>
            <div className="modal-body">
              <p style={{ fontSize: "13px", color: "#475569" }}>
                Are you sure you want to delete shelter <b>{deleteId}</b> from persistent SQLite storage?
              </p>
            </div>
            <div className="modal-footer">
              <button className="btn-outline" onClick={() => setDeleteId(null)}>Cancel</button>
              <button className="btn-danger" onClick={handleDeleteShelter}>Delete Permanently</button>
            </div>
          </div>
        </div>
      )}

      <Footer />
    </>
  );
}

/* ============================================================
   RESOURCE MANAGEMENT
   ============================================================ */

function ResourcePage({ resources = [], summary = null, onRefresh }) {
  const [filter, setFilter] = useState("All");
  const [search, setSearch] = useState("");
  const [modalOpen, setModalOpen] = useState(false);
  const [editingResource, setEditingResource] = useState(null);
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [resetConfirmOpen, setResetConfirmOpen] = useState(false);
  const [deleteId, setDeleteId] = useState(null);

  const [formData, setFormData] = useState({
    resource_id: "",
    resource_type: "",
    location: "",
    required_quantity: "",
    available_quantity: "",
    unit: "",
    notes: ""
  });
  const [formError, setFormError] = useState("");
  const [actionSuccess, setActionSuccess] = useState("");

  const filteredResources = resources.filter((r) => {
    if (search.trim()) {
      const q = search.toLowerCase();
      const tMatch = (r.resource_type || "").toLowerCase().includes(q);
      const lMatch = (r.location || "").toLowerCase().includes(q);
      const idMatch = (r.resource_id || "").toLowerCase().includes(q);
      if (!tMatch && !lMatch && !idMatch) return false;
    }
    if (filter === "All") return true;
    return r.status === filter;
  });

  function openCreateModal() {
    setEditingResource(null);
    setFormData({
      resource_id: "",
      resource_type: "",
      location: "",
      required_quantity: "",
      available_quantity: "",
      unit: "units",
      notes: ""
    });
    setFormError("");
    setModalOpen(true);
  }

  function openEditModal(resource) {
    setEditingResource(resource);
    setFormData({
      resource_id: resource.resource_id,
      resource_type: resource.resource_type,
      location: resource.location,
      required_quantity: resource.required_quantity,
      available_quantity: resource.available_quantity,
      unit: resource.unit,
      notes: resource.notes || ""
    });
    setFormError("");
    setModalOpen(true);
  }

  async function handleSaveResource(e) {
    e.preventDefault();
    setFormError("");

    const req = parseFloat(formData.required_quantity);
    const avail = parseFloat(formData.available_quantity);

    if (!formData.resource_type.trim() || !formData.location.trim() || !formData.unit.trim()) {
      setFormError("Resource type, location, and unit cannot be empty.");
      return;
    }
    if (isNaN(req) || req < 0) {
      setFormError("Required quantity must be a non-negative number.");
      return;
    }
    if (isNaN(avail) || avail < 0) {
      setFormError("Available quantity must be a non-negative number.");
      return;
    }

    try {
      let res;
      if (editingResource) {
        res = await fetch(`${API}/api/resources/${encodeURIComponent(editingResource.resource_id)}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(formData)
        });
      } else {
        res = await fetch(`${API}/api/resources`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(formData)
        });
      }

      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || "Failed to save resource record.");
      }

      setModalOpen(false);
      setActionSuccess(editingResource ? "Resource updated successfully!" : "Resource created successfully!");
      setTimeout(() => setActionSuccess(""), 4000);
      onRefresh();
    } catch (err) {
      setFormError(err.message);
    }
  }

  async function handleDeleteResource() {
    if (!deleteId) return;
    try {
      const res = await fetch(`${API}/api/resources/${encodeURIComponent(deleteId)}`, {
        method: "DELETE"
      });
      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || "Failed to delete resource.");
      }
      setDeleteId(null);
      setActionSuccess("Resource deleted successfully.");
      setTimeout(() => setActionSuccess(""), 4000);
      onRefresh();
    } catch (err) {
      alert("Error deleting resource: " + err.message);
    }
  }

  async function handleResetToDemo() {
    try {
      const res = await fetch(`${API}/api/resources/reset`, { method: "POST" });
      if (!res.ok) throw new Error("Failed to reset resources.");
      setResetConfirmOpen(false);
      setActionSuccess("Resources successfully reset to verified demo dataset.");
      setTimeout(() => setActionSuccess(""), 4000);
      onRefresh();
    } catch (err) {
      alert(err.message);
    }
  }

  async function handleImportCSV(e) {
    const file = e.target.files[0];
    if (!file) return;

    const body = new FormData();
    body.append("file", file);

    try {
      const res = await fetch(`${API}/api/resources/import`, {
        method: "POST",
        body
      });
      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || "Import failed.");
      }
      const data = await res.json();
      setImportModalOpen(false);
      setActionSuccess(`Successfully imported ${data.imported_count} resource records!`);
      setTimeout(() => setActionSuccess(""), 4000);
      onRefresh();
    } catch (err) {
      alert("Import error: " + err.message);
    }
  }

  return (
    <>
      <PageIntro
        tag="RESOURCE OPERATIONS"
        title="Emergency Resource Inventory"
        text="Track emergency supply inventories, required stockpiles, and dynamic shortages across relief sectors. Enforces non-negative balances and strict unit alignment."
      />

      {/* ACTION MESSAGES */}
      {actionSuccess && (
        <div className="alert-box alert-success" style={{ marginBottom: "16px" }}>
          ✓ {actionSuccess}
        </div>
      )}

      {/* METRIC CARDS */}
      <div className="stats-grid" style={{ marginBottom: "20px" }}>
        <StatCard
          label="Tracked Records"
          value={summary?.total_resources ?? resources.length}
          subtext="Persistent Inventory Items"
          icon="📦"
        />
        <StatCard
          label="Supply Categories"
          value={summary?.resource_types_tracked ?? 0}
          subtext="Distinct Commodities"
          icon="🏷️"
        />
        <StatCard
          label="Active Shortages"
          value={summary?.shortage_count ?? 0}
          danger={(summary?.shortage_count ?? 0) > 0}
          subtext="Required > Available"
          icon="🚨"
        />
        <StatCard
          label="Sufficient Stock"
          value={summary?.sufficient_count ?? 0}
          subtext="Balanced Supply Lines"
          icon="✓"
        />
        <StatCard
          label="Surplus Reserves"
          value={summary?.surplus_count ?? 0}
          subtext="Available > Required"
          icon="📈"
        />
      </div>

      {/* TOOLBAR */}
      <div className="toolbar-container">
        <div className="filters" style={{ margin: 0 }}>
          {["All", "Shortage", "Sufficient", "Surplus"].map((st) => (
            <button
              key={st}
              className={filter === st ? "filter active" : "filter"}
              onClick={() => setFilter(st)}
            >
              {st}
            </button>
          ))}
        </div>

        <div className="toolbar-actions">
          <input
            type="text"
            className="search-input"
            placeholder="Search resource, location, or unit..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <button className="btn-primary" onClick={openCreateModal}>
            + Add Resource
          </button>
          <a
            href={`${API}/api/resources/template`}
            download="resources_template.csv"
            className="btn-outline"
            style={{ textDecoration: "none" }}
          >
            📥 CSV Template
          </a>
          <button className="btn-outline" onClick={() => setImportModalOpen(true)}>
            📤 Import CSV
          </button>
          <button className="btn-outline" onClick={() => setResetConfirmOpen(true)}>
            ↻ Reset to Demo
          </button>
        </div>
      </div>

      {/* RESOURCE TABLE */}
      <Panel>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Resource ID & Type</th>
                <th>Location / Zone</th>
                <th>Required</th>
                <th>Available</th>
                <th>Shortage</th>
                <th>Surplus</th>
                <th>Status</th>
                <th>Data Source</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredResources.map((r) => (
                <tr key={r.resource_id} style={r.shortage > 0 ? { background: "#fffbeb" } : {}}>
                  <td>
                    <b>{r.resource_type}</b>
                    <small style={{ display: "block", color: "#64748b" }}>{r.resource_id}</small>
                    {r.notes && <small style={{ display: "block", color: "#94a3b8", fontStyle: "italic" }}>{r.notes}</small>}
                  </td>
                  <td>{r.location}</td>
                  <td><b>{r.required_quantity} {r.unit}</b></td>
                  <td>{r.available_quantity} {r.unit}</td>
                  <td>
                    <b style={{ color: r.shortage > 0 ? "#dc2626" : "#64748b" }}>
                      {r.shortage > 0 ? `${r.shortage} ${r.unit}` : "—"}
                    </b>
                  </td>
                  <td>
                    <b style={{ color: r.surplus > 0 ? "#16a34a" : "#64748b" }}>
                      {r.surplus > 0 ? `${r.surplus} ${r.unit}` : "—"}
                    </b>
                  </td>
                  <td>
                    <Badge text={r.status} />
                  </td>
                  <td>
                    <span className={`badge ${r.data_source && r.data_source.includes("DEMO") ? "demo-source" : "user-source"}`}>
                      {r.data_source && r.data_source.includes("DEMO") ? "DEMO DATA" : "USER SUPPLIED"}
                    </span>
                  </td>
                  <td>
                    <div style={{ display: "flex", gap: "6px" }}>
                      <button className="btn-outline btn-sm" onClick={() => openEditModal(r)}>Edit</button>
                      <button className="btn-danger btn-sm" onClick={() => setDeleteId(r.resource_id)}>Delete</button>
                    </div>
                  </td>
                </tr>
              ))}

              {filteredResources.length === 0 && (
                <tr>
                  <td colSpan={9} style={{ textAlign: "center", padding: "32px" }}>
                    No resource inventory records match the current criteria.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Panel>

      {/* CREATE / EDIT MODAL */}
      {modalOpen && (
        <div className="modal-backdrop" onClick={() => setModalOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{editingResource ? `Edit Resource (${editingResource.resource_id})` : "Add Emergency Resource"}</h3>
              <button className="modal-close" onClick={() => setModalOpen(false)}>✕</button>
            </div>
            <form onSubmit={handleSaveResource}>
              <div className="modal-body">
                {formError && <div className="alert-box alert-danger">{formError}</div>}

                <div className="form-group">
                  <label>Resource ID {editingResource ? "*" : "(Optional — auto-generated if left blank)"}</label>
                  <input
                    type="text"
                    className="form-input"
                    value={formData.resource_id}
                    onChange={(e) => setFormData({ ...formData, resource_id: e.target.value })}
                    placeholder="e.g. RES-05 (or leave blank to auto-generate)"
                    disabled={!!editingResource}
                  />
                  <small className="form-hint">
                    {editingResource
                      ? "Resource ID cannot be modified."
                      : "Optional unique identifier. A standard sequential ID (e.g. RES-05) will be assigned if left blank."}
                  </small>
                </div>

                <div className="form-group">
                  <label>Resource Commodity Type *</label>
                  <input
                    type="text"
                    className="form-input"
                    value={formData.resource_type}
                    onChange={(e) => setFormData({ ...formData, resource_type: e.target.value })}
                    placeholder="e.g. Drinking Water, Medical Trauma Kits, Emergency Cots"
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Location / Zone *</label>
                  <input
                    type="text"
                    className="form-input"
                    value={formData.location}
                    onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                    placeholder="e.g. San Juan, Metro Valley, Okanogan"
                    required
                  />
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "10px" }}>
                  <div className="form-group">
                    <label>Required Quantity *</label>
                    <input
                      type="number"
                      step="any"
                      min="0"
                      className="form-input"
                      value={formData.required_quantity}
                      onChange={(e) => setFormData({ ...formData, required_quantity: e.target.value })}
                      placeholder="e.g. 10000"
                      required
                    />
                  </div>

                  <div className="form-group">
                    <label>Available Stock *</label>
                    <input
                      type="number"
                      step="any"
                      min="0"
                      className="form-input"
                      value={formData.available_quantity}
                      onChange={(e) => setFormData({ ...formData, available_quantity: e.target.value })}
                      placeholder="e.g. 6000"
                      required
                    />
                  </div>

                  <div className="form-group">
                    <label>Unit of Measure *</label>
                    <input
                      type="text"
                      className="form-input"
                      value={formData.unit}
                      onChange={(e) => setFormData({ ...formData, unit: e.target.value })}
                      placeholder="Liters, kits, packs"
                      required
                    />
                  </div>
                </div>

                <div className="form-group">
                  <label>Operational Notes (Optional)</label>
                  <textarea
                    className="form-textarea"
                    value={formData.notes}
                    onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
                    placeholder="Convoy en route, staging warehouse A, refrigeration needed..."
                  />
                </div>
              </div>
              <div className="modal-footer">
                <button type="button" className="btn-outline" onClick={() => setModalOpen(false)}>Cancel</button>
                <button type="submit" className="btn-primary">{editingResource ? "Save Changes" : "Create Resource"}</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* CSV IMPORT MODAL */}
      {importModalOpen && (
        <div className="modal-backdrop" onClick={() => setImportModalOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Bulk Import Resources via CSV</h3>
              <button className="modal-close" onClick={() => setImportModalOpen(false)}>✕</button>
            </div>
            <div className="modal-body">
              <p style={{ fontSize: "13px", color: "#475569", marginBottom: "14px" }}>
                Upload a structured CSV file with columns: <code>resource_type</code>, <code>location</code>, <code>required_quantity</code>, <code>available_quantity</code>, <code>unit</code>.
              </p>
              <div className="form-group">
                <input
                  type="file"
                  accept=".csv"
                  className="form-input"
                  onChange={handleImportCSV}
                />
              </div>
              <small className="form-hint">
                Need the standard template? <a href={`${API}/api/resources/template`} download>Download resources_template.csv</a>
              </small>
            </div>
            <div className="modal-footer">
              <button className="btn-outline" onClick={() => setImportModalOpen(false)}>Cancel</button>
            </div>
          </div>
        </div>
      )}

      {/* RESET CONFIRMATION */}
      {resetConfirmOpen && (
        <div className="modal-backdrop" onClick={() => setResetConfirmOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Reset Emergency Resources to Demo?</h3>
              <button className="modal-close" onClick={() => setResetConfirmOpen(false)}>✕</button>
            </div>
            <div className="modal-body">
              <p style={{ fontSize: "13px", color: "#475569" }}>
                This action will restore the verified demo resource inventory (including shortages, sufficient stocks, and surpluses) and overwrite custom edits.
              </p>
              <div className="alert-box alert-warning" style={{ marginTop: "12px" }}>
                All seeded records are explicitly marked: <b>DEMO DATA — NOT LIVE MUNICIPAL INVENTORY</b>.
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn-outline" onClick={() => setResetConfirmOpen(false)}>Cancel</button>
              <button className="btn-danger" onClick={handleResetToDemo}>Confirm Reset</button>
            </div>
          </div>
        </div>
      )}

      {/* DELETE CONFIRMATION */}
      {deleteId && (
        <div className="modal-backdrop" onClick={() => setDeleteId(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Confirm Resource Deletion</h3>
              <button className="modal-close" onClick={() => setDeleteId(null)}>✕</button>
            </div>
            <div className="modal-body">
              <p style={{ fontSize: "13px", color: "#475569" }}>
                Are you sure you want to delete resource <b>{deleteId}</b> from persistent SQLite storage?
              </p>
            </div>
            <div className="modal-footer">
              <button className="btn-outline" onClick={() => setDeleteId(null)}>Cancel</button>
              <button className="btn-danger" onClick={handleDeleteResource}>Delete Permanently</button>
            </div>
          </div>
        </div>
      )}

      <Footer />
    </>
  );
}

/* ============================================================
   ANALYTICS PAGE
   ============================================================ */

function AnalyticsPage({ data, shelters = [], shelterSummary = null, resources = [], resourceSummary = null }) {
  const incidents = data.incidents || [];
  const anomalies = data.anomalies || [];

  // Impact Tiers
  const tierCounts = incidents.reduce((acc, i) => {
    const tier = i.impact_tier || "Unassessed";
    acc[tier] = (acc[tier] || 0) + 1;
    return acc;
  }, {});
  const maxTierCount = Math.max(1, ...Object.values(tierCounts));

  // Disaster Types
  const typeCounts = incidents.reduce((counts, incident) => {
    const type = String(incident.type || "Unknown").trim() || "Unknown";
    counts[type] = (counts[type] || 0) + 1;
    return counts;
  }, {});
  const sortedTypes = Object.entries(typeCounts).sort((a, b) => b[1] - a[1]);
  const maxTypeCount = Math.max(1, ...sortedTypes.map(([, count]) => count));

  // Economic Loss Stats
  const summary = data.summary || {};
  const lossStats = computeLossCoverage(incidents);
  const assessedLoss = incidents.filter(i => isValidLoss(i.estimated_loss));
  const totalAssessedDamage = assessedLoss.reduce((sum, i) => sum + Number(i.estimated_loss), 0);
  const avgLoss = assessedLoss.length > 0 ? Math.round(totalAssessedDamage / assessedLoss.length) : 0;
  const coveragePct = lossStats.coveragePct;

  return (
    <>
      <PageIntro
        tag="INTELLIGENCE & ANALYTICS"
        title="Disaster Intelligence & Inventory Analytics"
        text="Comprehensive multi-attribute analytics comparing rule-based incident impact tiers, meteorological hazard profiles, and live SQLite relief capacity."
      />

      <div className="dashboard-grid">
        {/* RULE-BASED IMPACT TIER DISTRIBUTION */}
        <Panel title="Rule-Based Impact Tier Distribution">
          <div className="analytics-list">
            {Object.entries(tierCounts).map(([tier, count]) => (
              <div className="analytics-item" key={tier}>
                <div className="row-header">
                  <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <Badge text={tier} />
                    <span style={{ fontSize: "12px", color: "#64748b" }}>
                      ({incidents.length > 0 ? ((count / incidents.length) * 100).toFixed(1) : 0}%)
                    </span>
                  </div>
                  <strong>{count.toLocaleString()} incidents</strong>
                </div>
                <div className="progress-bar">
                  <span
                    style={{
                      width: `${(count / maxTierCount) * 100}%`,
                      background: tier.includes("Critical") ? "#ef4444" : tier.includes("High") ? "#f59e0b" : tier.includes("Moderate") ? "#3b82f6" : "#64748b"
                    }}
                  ></span>
                </div>
              </div>
            ))}
          </div>
          <small style={{ display: "block", marginTop: "10px", color: "#64748b" }}>
            * Evaluated deterministically from recorded fatalities, injuries, and property/crop damage values.
          </small>
        </Panel>

        {/* DISASTER TYPE DISTRIBUTION */}
        <Panel title="Disaster Hazard Distribution">
          {sortedTypes.length === 0 ? (
            <div className="analytics-item"><b>No incident records available</b></div>
          ) : (
            <div className="analytics-list">
              {sortedTypes.slice(0, 8).map(([type, count]) => (
                <div className="analytics-item" key={type}>
                  <div className="row-header">
                    <b>{type}</b>
                    <strong>{count.toLocaleString()}</strong>
                  </div>
                  <div className="progress-bar">
                    <span style={{ width: `${(count / maxTypeCount) * 100}%` }}></span>
                  </div>
                </div>
              ))}
            </div>
          )}
          <small style={{ display: "block", marginTop: "10px", color: "#64748b" }}>
            Counts reflect meteorological storm event types received in current active assessment.
          </small>
        </Panel>

        {/* ECONOMIC LOSS COVERAGE & DAMAGE */}
        <Panel title="Economic Damage & Loss Coverage">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", marginBottom: "14px" }}>
            <div style={{ background: "#f8fafc", padding: "12px", borderRadius: "8px", border: "1px solid #e2e8f0" }}>
              <small style={{ color: "#64748b", display: "block" }}>Assessed Loss Coverage</small>
              <b style={{ fontSize: "18px", color: "#0f172a" }}>{lossStats.coveragePct}%</b>
              <small style={{ display: "block", color: "#22c55e", marginTop: "2px" }}>
                {lossStats.assessed} / {lossStats.total} Records Assessed ({lossStats.unassessed} Unassessed)
              </small>
            </div>
            <div style={{ background: "#f8fafc", padding: "12px", borderRadius: "8px", border: "1px solid #e2e8f0" }}>
              <small style={{ color: "#64748b", display: "block" }}>Total Evaluated Loss</small>
              <b style={{ fontSize: "18px", color: "#0f172a" }}>${totalAssessedDamage.toLocaleString()}</b>
              <small style={{ display: "block", color: "#64748b", marginTop: "2px" }}>
                Avg: ${avgLoss.toLocaleString()} / event
              </small>
            </div>
          </div>
          <div className="alert-box alert-warning" style={{ marginBottom: "10px" }}>
            ℹ️ <b>Missing Data Protocol:</b> {lossStats.unassessed} unassessed/null loss records are treated strictly as unassessed, not verified zero damage.
          </div>
          <div style={{ fontSize: "11px", color: "#64748b", padding: "6px 10px", background: "#f1f5f9", borderRadius: "6px" }}>
            Active Dataset Provenance: <b>{data.integratedDataset?.filename || summary.filename || "Active Assessment"}</b> · <b>{lossStats.total}</b> records · Job ID: <code>{summary.job_id || "Active"}</code> · Assessment ID: <code>{summary.assessment_id || "Active"}</code>
          </div>
        </Panel>

        {/* SHELTER UTILIZATION & CAPACITY */}
        <Panel title="Shelter Utilization & Available Beds">
          {shelters.length === 0 ? (
            <div className="analytics-item">
              <b>No Shelter Facilities Stored</b>
              <small>Add shelter records in Shelter Management to track capacities.</small>
            </div>
          ) : (
            <div className="analytics-list">
              {shelters.slice(0, 5).map((s) => (
                <div className="analytics-item" key={s.shelter_id}>
                  <div className="row-header">
                    <div>
                      <b>{s.shelter_name}</b>
                      <small>{s.location} · {s.available_beds} beds available</small>
                    </div>
                    <Badge text={s.status} />
                  </div>
                  <div className="progress-bar">
                    <span
                      style={{
                        width: `${Math.min(s.utilization_percentage || 0, 100)}%`,
                        background: s.status === "Full" ? "#ef4444" : s.status === "Near Capacity" ? "#f59e0b" : "#22c55e"
                      }}
                    ></span>
                  </div>
                </div>
              ))}
            </div>
          )}
          <small style={{ display: "block", marginTop: "10px", color: "#64748b" }}>
            Persistent SQLite inventory: <b>{shelterSummary?.total_capacity?.toLocaleString() || 0}</b> total capacity, <b>{shelterSummary?.total_occupancy?.toLocaleString() || 0}</b> occupied.
          </small>
        </Panel>
      </div>

      {/* ISOLATION FOREST ANOMALY DETECTION */}
      <Panel title="Isolation Forest Multidimensional Anomaly Outliers">
        <div style={{ marginBottom: "14px", padding: "10px 14px", background: "rgba(255,255,255,0.03)", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.06)", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "8px" }}>
          <div>
            <b>Model: Isolation Forest (Trained on 72,360 Historical NOAA Storm Records)</b>
            <small style={{ display: "block", color: "var(--text-muted, #94a3b8)", marginTop: "2px" }}>
              Assessed Records: <strong>{incidents.length}</strong> · Flagged Outliers: <strong>{anomalies.length}</strong> ({incidents.length > 0 ? ((anomalies.length / incidents.length) * 100).toFixed(1) : 0}%)
            </small>
          </div>
          <Badge text="Isolation Forest" />
        </div>

        <div className="anomaly-context-banner">
          <small>
            <b>Statistical Outlier Review:</b> Flagged records differ from multidimensional historical patterns learned by Isolation Forest. An unusual record represents atypical incident telemetry (e.g. rare coordinates or high damage) requiring human verification, not automatic emergency declaration.
          </small>
        </div>

        <div className="analytics-list">
          {anomalies.length === 0 ? (
            <div className="analytics-item">
              <b>No anomaly data available</b>
              <small>Isolation Forest has not returned any flagged records.</small>
            </div>
          ) : (
            anomalies.slice(0, 8).map((item, index) => (
              <div className="analytics-item" key={index}>
                <div className="row-header">
                  <div>
                    <b>{item.id ? `${item.id} · ${item.type || 'Incident'}` : item.shelter}</b>
                    <small style={{ color: "#64748b" }}>
                      {item.zone}
                      {item.contributing_evidence ? ` · ${item.contributing_evidence}` : ""}
                    </small>
                  </div>
                  <strong>{item.anomaly ? "⚠ Unusual" : "✓ Normal"}</strong>
                </div>
                <div className="progress-info">
                  <span>Isolation Forest Multi-Attribute Score</span>
                  <b>{item.anomaly_score != null ? Number(item.anomaly_score).toFixed(4) : "—"}</b>
                </div>
              </div>
            ))
          )}
        </div>
      </Panel>

      <Footer />
    </>
  );
}

/* ============================================================
   RECOMMENDATIONS PAGE
   ============================================================ */

function RecommendationPage({ recommendations = [], onMatchInventory }) {
  return (
    <>
      <PageIntro
        tag="DECISION SUPPORT"
        title="Recommended Response Actions"
        text="Actionable operational suggestions dynamically generated from active incident evidence and persistent shelter/resource balances. Distinguishes deterministic screening, anomaly investigation, and logistics requests."
      />

      <div className="recommendation-page">
        <div className="alert-box alert-info" style={{ marginBottom: "16px" }}>
          ℹ️ <b>Operational Decision Support:</b> Recommendations are advisory decision-support outputs based on deterministic impact screening and statistical anomaly triage of historical NOAA event records. Recommendations do not represent live municipal telemetry, official CAD dispatch approval, or automatic resource dispatch. Relief inventories reflect demo baseline quantities unless overridden with verified municipal data. Full shelters (100% capacity) are strictly excluded from evacuation intake.
        </div>

        {recommendations.some((item) => item.source === "integrated") && (
          <div className="analytics-item" style={{ marginBottom: "14px", borderLeft: "3px solid #2563eb" }}>
            <b>Integrated Evidence-Based Suggestions</b>
            <small style={{ display: "block", marginTop: "4px" }}>
              These suggestions are generated dynamically from current incident casualty/damage fields and persistent facility inventories. Full shelters (100% capacity) are never recommended for intake. Confirm all details with authorized dispatchers.
            </small>
          </div>
        )}

        {recommendations.map((item, index) => (
          <div className="recommendation-card" key={index}>
            <div className="recommendation-number">{index + 1}</div>

            <div style={{ flex: 1 }}>
              <div className="recommendation-top">
                <div>
                  <b style={{ fontSize: "14px" }}>{item.zone || item.location}</b>
                  {item.incident_id && (
                    <small style={{ marginLeft: "8px", color: "var(--text-muted, #94a3b8)" }}>
                      <code>{item.incident_id}</code> · {item.disaster_type || "Incident"}
                    </small>
                  )}
                </div>

                <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
                  {item.guidance_type && (
                    <span className="badge" style={{ fontSize: "11px", padding: "2px 8px" }}>
                      {item.guidance_type}
                    </span>
                  )}
                  {item.impact_tier && (
                    <Badge text={item.impact_tier} />
                  )}
                </div>
              </div>

              {item.supporting_evidence && (
                <div style={{ marginTop: "6px", fontSize: "13px", color: "#334155" }}>
                  <b>Supporting Evidence:</b> {item.supporting_evidence}
                </div>
              )}

              <p style={{ marginTop: "6px", fontSize: "13px", lineHeight: "1.5" }}>
                <b>Proposed Action:</b> {item.action}
              </p>

              {item.reason && (
                <div style={{ marginTop: "6px", fontSize: "12px", color: "#64748b", background: "#f8fafc", padding: "6px 10px", borderRadius: "4px", border: "1px solid #e2e8f0" }}>
                  <b>Operational Rationale:</b> {item.reason}
                </div>
              )}

              {/* Matched Shelter Guardrail Notice */}
              {item.matched_shelter && (
                <div style={{ marginTop: "8px", fontSize: "11px", color: item.matched_shelter.status === "Full" ? "#b91c1c" : "#1e40af" }}>
                  Facility Context: <b>{item.matched_shelter.shelter_name}</b> ({item.matched_shelter.available_beds} beds available, {item.matched_shelter.status})
                </div>
              )}

              <div style={{ marginTop: "10px", display: "flex", justifyContent: "flex-end" }}>
                <button
                  className="btn-outline btn-sm"
                  onClick={() => onMatchInventory({
                    id: item.incident_id,
                    type: item.disaster_type,
                    zone: item.zone || item.location,
                    impact_tier: item.impact_tier,
                    contributing_evidence: item.supporting_evidence
                  })}
                >
                  📦 Inspect Local Inventory
                </button>
              </div>
            </div>
          </div>
        ))}

        {recommendations.length === 0 && (
          <div className="analytics-item">
            <b>No recommendations generated</b>
            <small>Upload an incident dataset to generate operational decision-support recommendations.</small>
          </div>
        )}
      </div>

      <Footer />
    </>
  );
}

/* ============================================================
   REUSABLE COMPONENTS
   ============================================================ */

function Panel({ title, action, onClick, children }) {
  return (
    <section className="panel">
      {title && (
        <div className="panel-header">
          <h3>{title}</h3>
          {action && (
            <button onClick={onClick}>
              {action} →
            </button>
          )}
        </div>
      )}
      {children}
    </section>
  );
}

function PageIntro({ tag, title, text }) {
  return (
    <section className="page-intro">
      <div>
        <div className="eyebrow">{tag}</div>
        <h2>{title}</h2>
        <p>{text}</p>
      </div>
      <div className="page-intro-icon">✦</div>
    </section>
  );
}

function Footer() {
  return (
    <footer>
      ResQ Intelligence · Member 2 · React + FastAPI + SQLite Persistence
    </footer>
  );
}

export default App;