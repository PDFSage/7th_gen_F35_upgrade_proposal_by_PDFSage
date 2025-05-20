#!/usr/bin/env python
# Minimal comments per user request
from datetime import timezone

# Optional imports with safe fall-backs
try:
    import requests
except ImportError:  # requests stub
    class _RequestsStub:
        def __getattr__(self, _): raise RuntimeError("requests library not available")
    requests = _RequestsStub()

try:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:  # PIL stub
    class _PILStub:
        def __getattr__(self, _): raise RuntimeError("PIL not available")
    Image = _PILStub()
    class _PILImageFileStub: LOAD_TRUNCATED_IMAGES = False
    ImageFile = _PILImageFileStub()

try:
    from numpy import array
except ImportError:  # numpy stub
    def array(x, *_, **__): return x

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, get_origin, get_args

try:
    import psutil
except ImportError:
    psutil = None
try:
    import GPUtil
except ImportError:
    GPUtil = None

# OpenGL safe import / stubs
try:
    from OpenGL.GL import (
        glGenTextures, glBindTexture, glTexImage2D, glTexParameterf, glEnable, glDisable,
        glBegin, glEnd, glVertex3f, glColor3f, glPointSize, glClear,
        glClearColor, glMatrixMode, glLoadIdentity, glPushMatrix, glPopMatrix,
        glRasterPos2f, glRasterPos3f, GL_TEXTURE_2D, GL_RGB, GL_UNSIGNED_BYTE,
        GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR, GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT, GL_PROJECTION, GL_MODELVIEW, GL_POINTS, GL_DEPTH_TEST
    )
    from OpenGL.GLU import (
        gluPerspective, gluLookAt, gluNewQuadric, gluQuadricTexture,
        gluSphere, gluOrtho2D
    )
    from OpenGL.GLUT import (
        glutBitmapCharacter, GLUT_BITMAP_HELVETICA_10, GLUT_BITMAP_HELVETICA_12,
        glutInit, glutInitDisplayMode, glutInitWindowSize, glutCreateWindow,
        glutDisplayFunc, glutIdleFunc, glutKeyboardFunc, glutMainLoop, glutSwapBuffers,
        glutLeaveMainLoop, glutPostRedisplay,
        GLUT_DOUBLE, GLUT_RGB, GLUT_DEPTH
    )
    GL_TRUE = 1
except Exception as e:
    logging.warning(f'OpenGL import failed: {e}')
    def _stub(*_, **__): pass
    glGenTextures = lambda n=1: 0
    glBindTexture = glTexImage2D = glTexParameterf = glEnable = glDisable = _stub
    glBegin = glEnd = glVertex3f = glColor3f = glPointSize = _stub
    glClear = glClearColor = glMatrixMode = glLoadIdentity = glPushMatrix = glPopMatrix = _stub
    glRasterPos2f = glRasterPos3f = _stub
    GL_TEXTURE_2D = GL_RGB = GL_UNSIGNED_BYTE = GL_TEXTURE_MIN_FILTER = GL_TEXTURE_MAG_FILTER = GL_LINEAR = 0
    GL_COLOR_BUFFER_BIT = GL_DEPTH_BUFFER_BIT = GL_PROJECTION = GL_MODELVIEW = GL_POINTS = GL_DEPTH_TEST = 0
    GL_TRUE = 1
    def gluPerspective(*_, **__): pass
    def gluLookAt(*_, **__): pass
    def gluNewQuadric(): return None
    def gluQuadricTexture(*_, **__): pass
    def gluSphere(*_, **__): pass
    def gluOrtho2D(*_, **__): pass
    def glutBitmapCharacter(*_, **__): pass
    GLUT_BITMAP_HELVETICA_10 = GLUT_BITMAP_HELVETICA_12 = None
    def glutInit(*_): pass
    def glutInitDisplayMode(*_): pass
    def glutInitWindowSize(*_, **__): pass
    def glutCreateWindow(*_): pass
    def glutDisplayFunc(*_): pass
    def glutIdleFunc(*_): pass
    def glutKeyboardFunc(*_): pass
    def glutMainLoop(): pass
    def glutSwapBuffers(): pass
    def glutLeaveMainLoop(): pass
    def glutPostRedisplay(): pass
    GLUT_DOUBLE = GLUT_RGB = GLUT_DEPTH = 0

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

G = 9.81
RAD = math.pi / 180.0
FT_PER_M = 3.28084
MPH_PER_KMH = 0.621371
FIRE_RANGE = 30000.0
EARTH_RADIUS_M = 6371000.0

AIRPORTS = {'KIAH': (29.9902, -95.3368), 'KSFO': (37.6189, -122.3750)}
RUNWAYS = {'KIAH': [{'hdg': 150.0}], 'KSFO': [{'hdg': 100.0}]}

GOOGLE_MAP_TEXTURE = 'google_map.jpg'
GOOGLE_MAP_URL = ''
EARTH_TEXTURE = 'earth.jpg'
EARTH_URLS = []

def ensure_texture(_tex, _url): return False  # external source disabled

def _ensure_defaults(obj):
    if obj is None: return
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            if v is None:
                if f.default is not dataclasses.MISSING:
                    setattr(obj, f.name, copy.deepcopy(f.default))
                elif f.default_factory is not dataclasses.MISSING:
                    setattr(obj, f.name, f.default_factory())
                else:
                    t = f.type
                    if get_origin(t) is Optional: t = get_args(t)[0]
                    if dataclasses.is_dataclass(t): setattr(obj, f.name, t())
                    elif t is float: setattr(obj, f.name, 0.0)
                    elif t is int: setattr(obj, f.name, 0)
                    elif t is bool: setattr(obj, f.name, False)
                    elif t is str: setattr(obj, f.name, "")
                    elif t in (list, dict): setattr(obj, f.name, t())
                    else: setattr(obj, f.name, None)
                v = getattr(obj, f.name)
            _ensure_defaults(v)
    elif isinstance(obj, list):
        for item in obj: _ensure_defaults(item)
    elif isinstance(obj, dict):
        for item in obj.values(): _ensure_defaults(item)

class RealisticWeather:
    def __init__(self, ws=None, wd=None, seed=1):
        random.seed(seed)
        self.ws = ws if ws is not None else random.uniform(0, 20)
        self.wd = wd if wd is not None else random.uniform(0, 360)

class _SparseNN:
    def __init__(self, threshold: float = 0.01): self.threshold = threshold
    def map(self, _k: str, value: Any) -> Any:
        if value is None: return 1.0
        if isinstance(value, (int, float)) and not math.isfinite(value): return 1.0
        return value

class _InitMixin:
    def initialize(self, data: Dict[str, Any], snn: Optional[_SparseNN] = None):
        snn = snn or _SparseNN()
        for k, v in (data or {}).items():
            if hasattr(self, k): setattr(self, k, snn.map(k, v))
    def __post_init__(self):
        self.initialize({})
        _ensure_defaults(self)

@dataclass
class GeoSpatialState(_InitMixin):
    lat: float = 0.0
    lon: float = 0.0
    alt_m: float = 0.0
    speed_mps: float = 0.0

@dataclass
class OrientationState(_InitMixin):
    phi: float = 0.0
    theta: float = 0.0
    psi: float = 0.0
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0

@dataclass
class PhysicalState(_InitMixin):
    u: float = 0.0
    v: float = 0.0
    w: float = 0.0
    acceleration: float = 0.0

@dataclass
class MaterialState(_InitMixin):
    temperature: float = 15.0
    pressure: float = 0.0
    humidity: float = 0.0
    integrity: float = 1.0

@dataclass
class DecisionMakingState(_InitMixin):
    mode: str = "IDLE"
    target: Optional[str] = None
    threat_level: float = 0.0

@dataclass
class ElectronicState(_InitMixin):
    cpu_load: float = 0.0
    gpu_load: float = 0.0
    battery_pct: float = 1.0
    network_latency_ms: float = 0.0

@dataclass
class AlgorithmicState(_InitMixin):
    model_name: str = ""
    version: str = ""
    confidence: float = 0.0
    loss: float = 0.0

@dataclass
class MagneticFieldState(_InitMixin):
    Bx: float = 0.0
    By: float = 0.0
    Bz: float = 0.0

@dataclass
class EMRadiationState(_InitMixin):
    e_field_vpm: float = 0.0
    power_density_wpm2: float = 0.0
    frequency_hz: float = 0.0

@dataclass
class RadarSignatureState(_InitMixin):
    rcs_m2: float = 0.0
    detection_range_m: float = 0.0
    cone_deg: float = 60.0
    detection_coeff: float = 1.0
    active: bool = True

@dataclass
class NeuralNetworkState(_InitMixin):
    layers: List[str] = field(default_factory=list)
    sparse_activation: bool = True
    threshold: float = 0.01

@dataclass
class FlightControlState(_InitMixin):
    speed: float = 0.0
    throttle: float = 0.0
    on_ground: bool = True
    destroyed: bool = False
    fuel_mass: float = 15000.0
    fuel_capacity: float = 15000.0
    missile_total: int = 0
    missile_ready: int = 0

@dataclass
class SimulationState(_InitMixin):
    elapsed: float = 0.0
    sim_speed: float = 1.0

@dataclass
class EntityState(_InitMixin):
    geospatial: GeoSpatialState = field(default_factory=GeoSpatialState)
    orientation: OrientationState = field(default_factory=OrientationState)
    physical: PhysicalState = field(default_factory=PhysicalState)
    material: MaterialState = field(default_factory=MaterialState)
    decision: DecisionMakingState = field(default_factory=DecisionMakingState)
    electronic: ElectronicState = field(default_factory=ElectronicState)
    algorithmic: AlgorithmicState = field(default_factory=AlgorithmicState)
    magnetic: MagneticFieldState = field(default_factory=MagneticFieldState)
    em: EMRadiationState = field(default_factory=EMRadiationState)
    radar: RadarSignatureState = field(default_factory=RadarSignatureState)
    nn: NeuralNetworkState = field(default_factory=NeuralNetworkState)
    flight: FlightControlState = field(default_factory=FlightControlState)
    def initialize(self, data: Dict[str, Any], snn: Optional[_SparseNN] = None):
        snn = snn or _SparseNN()
        for k, v in (data or {}).items():
            if hasattr(self, k):
                attr = getattr(self, k)
                if dataclasses.is_dataclass(attr): attr.initialize(v, snn)
                else: setattr(self, k, snn.map(k, v))
    def _step_generic(self, parent, dt):
        if self.flight.destroyed: return
        accel = getattr(parent, 'pilot_input', PilotInput()).throttle * 30.0 - self.flight.throttle * 10.0
        self.physical.acceleration = accel
        self.flight.speed = max(0.0, self.flight.speed + accel * dt)
        dist = self.flight.speed * dt
        hdg = self.orientation.psi
        dlat = (dist / 111320.0) * math.cos(hdg)
        dlon = (dist / (111320.0 * max(0.01, math.cos(math.radians(self.geospatial.lat))))) * math.sin(hdg)
        self.geospatial.lat += dlat
        self.geospatial.lon += dlon
        climb = getattr(parent, "climb_rate", 0.0)
        self.geospatial.alt_m = max(0.0, self.geospatial.alt_m + climb * dt * (0 if self.flight.on_ground else 1))
    def step_aircraft(self, parent, dt): self._step_generic(parent, dt)
    def step_car(self, parent, dt): self._step_generic(parent, dt)
    def step_orbit(self, parent, dt):
        t = (UNIVERSE.simulation.elapsed + getattr(parent, "phase", 0.0)) % parent.orbit_period
        ang = 2 * math.pi * t / parent.orbit_period
        self.geospatial.lat = 0.0
        self.geospatial.lon = math.degrees(ang)
        self.geospatial.alt_m = parent.orbit_alt_km * 1000.0
        self.flight.on_ground = False
    def step_launch(self, parent, dt):
        if self.flight.destroyed: return
        if parent.launch_phase == 0:
            self.physical.acceleration = 15.0 * G
            self.flight.speed += self.physical.acceleration * dt
            self.geospatial.alt_m += self.flight.speed * dt
            if self.geospatial.alt_m >= 80000.0: parent.launch_phase = 1
        else:
            self.physical.acceleration = 0.0
            self.flight.speed = max(0.0, self.flight.speed - G * dt)
    def euler_deg(self):
        return (self.orientation.theta / RAD,
                self.orientation.phi / RAD,
                ((self.orientation.psi + math.pi) % (2 * math.pi) - math.pi) / RAD)

State6DOF = EntityState

@dataclass
class UniversalBo(_InitMixin):
    timestamp: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(timezone.utc))
    simulation: SimulationState = field(default_factory=SimulationState)
    entities: Dict[str, EntityState] = field(default_factory=dict)
    def add_entity(self, name: str, state: Optional[EntityState] = None) -> EntityState:
        if name not in self.entities: self.entities[name] = state or EntityState()
        self.entities[name].initialize({})
        self.validate_required_fields()
        return self.entities[name]
    def get(self, name: str) -> Optional[EntityState]:
        return self.entities.get(name)
    def validate_required_fields(self):
        for e in self.entities.values(): _ensure_defaults(e)

UNIVERSE = UniversalBo()

def calc_heading(orig_lat, orig_lon, dest_lat, dest_lon):
    dLon = math.radians(dest_lon - orig_lon)
    y = math.sin(dLon) * math.cos(math.radians(dest_lat))
    x = math.cos(math.radians(orig_lat)) * math.sin(math.radians(dest_lat)) - math.sin(math.radians(orig_lat)) * math.cos(math.radians(dest_lat)) * math.cos(dLon)
    return (math.atan2(y, x) + 2 * math.pi) % (2 * math.pi)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def magnetic_variation(lat, lon): return 10.0 * math.sin(math.radians(lon)) * math.cos(math.radians(lat))
def geo_to_xyz(lat, lon, radius=1.0):
    lat_r, lon_r = math.radians(lat), math.radians(lon)
    x = radius * math.cos(lat_r) * math.cos(lon_r)
    y = radius * math.sin(lat_r)
    z = radius * math.cos(lat_r) * math.sin(lon_r)
    return x, y, z
def runway_designation(hdg): return int(round(hdg / 10.0)) % 36 or 36

class Sensors:
    def __init__(self): self.aoa_L = self.aoa_R = self.mach = self.flap = self.on_gnd = self.fail_L = False
    def set_env(self, base_aoa, mach, flap, on_gnd):
        self.aoa_L = self.aoa_R = base_aoa
        self.mach, self.flap, self.on_gnd = mach, flap, on_gnd
    def aoa_status(self): return self.fail_L
    def avg_aoa(self): return 0.5 * (self.aoa_L + self.aoa_R)

@dataclass
class PilotInput:
    throttle: float = 0.0
    yoke: float = 0.0
    roll: float = 0.0
    rudder: float = 0.0
    trim: float = 0.0
    flap_cmd: int = 0
    gear_cmd: bool = False

@dataclass
class APCommand:
    yoke: float = 0.0
    roll: float = 0.0

class Autopilot:
    def __init__(self):
        self.tgt_alt = 0.0
        self.tgt_hdg = 0.0
    def set_targets(self, alt, hdg): self.tgt_alt, self.tgt_hdg = alt, hdg
    def command(self, _dt, _state): return APCommand()

class PilotModel:
    def decide(self, ac, flights, missiles, tankers):
        if ac.state.flight.on_ground:
            if not getattr(ac, 'takeoff_authorized', False):
                if ac.state.flight.speed < 0.5: return (0.0, ac.runway_hdg, 0.0)
                return (0.0, ac.runway_hdg, 0.2)
            return (1000.0, ac.runway_hdg, 1.0)
        inbound = [m for m in missiles if m['target'] is ac and m['dist'] < 8000.0]
        if inbound:
            ev_dir = (ac.state.orientation.psi + math.pi / 2) % (2 * math.pi)
            return (15000.0, ev_dir, 1.0)
        enemies = [f for f in flights if f.team != ac.team and isinstance(f, Fighter) and not f.state.flight.destroyed]
        if enemies:
            target = min(enemies, key=lambda t: haversine(ac.state.geospatial.lat, ac.state.geospatial.lon, t.state.geospatial.lat, t.state.geospatial.lon))
            hdg = calc_heading(ac.state.geospatial.lat, ac.state.geospatial.lon, target.state.geospatial.lat, target.state.geospatial.lon)
            return (12000.0, hdg, 1.0)
        return (8000.0, ac.state.orientation.psi, 0.8)

class BaseAgent:
    DEFAULT_ATTRS_SIMPLE = {'climb_rate': 0.0, 'turn_rate': 0.0, 'takeoff_authorized': False, 'runway_hdg': 0.0}
    def __init__(self, name: str):
        self.name = name
        self.state = UNIVERSE.add_entity(name)
        if not hasattr(self, 'pilot_input'): self.pilot_input = PilotInput()
        for attr, default in BaseAgent.DEFAULT_ATTRS_SIMPLE.items():
            if not hasattr(self, attr): setattr(self, attr, default)
        _ensure_defaults(self)
        _ensure_defaults(self.state)

class Missile(BaseAgent):
    def __init__(self, name, shooter=None, missile_type='GENERIC'):
        super().__init__(name)
        self.missile_type = missile_type
        self.active = False
        self.speed = 800.0
        self.shooter = shooter
        self.status = 'READY'
        if shooter is not None:
            self.state.geospatial.lat = shooter.state.geospatial.lat
            self.state.geospatial.lon = shooter.state.geospatial.lon
            self.state.geospatial.alt_m = shooter.state.geospatial.alt_m
            self.state.orientation.psi = shooter.state.orientation.psi
            self.state.flight.speed = shooter.state.flight.speed
            self.state.flight.on_ground = shooter.state.flight.on_ground
        else:
            self.state.flight.speed = self.speed
            self.state.flight.on_ground = True
        _ensure_defaults(self.state)
    def launch(self):
        self.active = True
        self.status = 'IN_FLIGHT'
    def step(self, dt):
        if not self.active: return
        self.state.geospatial.alt_m = max(0.0, self.state.geospatial.alt_m + 200.0 * dt)

class Aircraft(BaseAgent):
    def __init__(self, callsign='AC', team='EAST'):
        super().__init__(callsign)
        self.callsign = callsign
        self.ap = Autopilot()
        self.pilot = PilotModel()
        self.team = team
        self.dest = None
        self.rtb = False
        self.climb_rate = getattr(self, 'climb_rate', 50.0)
        self.turn_rate = getattr(self, 'turn_rate', math.radians(10.0))
        self.missiles: List[Missile] = []
        self.prev_psi = 0.0
        self.route = []
        self.route_idx = 0
        self.runway_hdg = getattr(self, 'runway_hdg', 0.0)
        self.pilot_name = ""
        self.missile_type = 'GENERIC'
        self.takeoff_authorized = getattr(self, 'takeoff_authorized', False)
        self.liftoff_speed = 80.0
        self.pilot_input = PilotInput()
        self.v1 = self.liftoff_speed * 0.7
        self.v2 = self.liftoff_speed * 0.9
        self.v1_time: Optional[float] = None
        self.v2_time: Optional[float] = None
    def step(self, dt): self.state.step_aircraft(self, dt)
    def load_missiles(self, count, missile_type=None):
        self.missiles = []
        m_type = missile_type if missile_type else self.missile_type
        for i in range(1, count + 1):
            self.missiles.append(Missile(f'{self.callsign}_MSL{i}', self, m_type))
        self.state.flight.missile_total = count
        self.state.flight.missile_ready = count
    def fire_missile(self):
        for m in self.missiles:
            if m.status == 'READY':
                m.launch()
                self.state.flight.missile_ready -= 1
                return m
        return None

class Fighter(Aircraft): pass
class Refueler(Aircraft): pass
class Car(Aircraft):
    def step(self, dt): self.state.step_car(self, dt)
class Airliner(Aircraft): pass
class TeslaCar(Car):
    def __init__(self, callsign, driver_name, route):
        super().__init__(callsign, 'CIV')
        self.driver_name = driver_name
        self.route = route
        self.coverage = True
        self.speed = 30.0 / 3.6

class F35(Fighter): pass
class F22(Fighter): pass
class KC46(Refueler): pass
class Boeing737Max(Airliner):
    def __init__(self, callsign='B737', team='CIV'):
        super().__init__(callsign, team)
        self.mcas = type('MCAS', (), {'cmd_count': 0, 'total_cmd': 0.0})()
        self.mcas_reported = False
class Boeing737MaxV1(Boeing737Max): pass
class Boeing737MaxV2(Boeing737Max): pass
class OrbitalAsset(Aircraft):
    def __init__(self, callsign='ORB', alt_km=420):
        super().__init__(callsign, 'SPACE')
        self.orbit_alt_km = alt_km
        self.orbit_radius = 6371 + alt_km
        self.orbit_period = 92 * 60
        self.phase = random.random() * 2 * math.pi
        self.state.flight.on_ground = False
    def step(self, dt): self.state.step_orbit(self, dt)

class ISS(OrbitalAsset):
    def __init__(self): super().__init__('ISS', 415)
class StarlinkSatellite(OrbitalAsset):
    def __init__(self, idx):
        super().__init__(f'STAR{idx}', 550)
        self.phase = idx * 30 * RAD

class LaunchVehicle(Aircraft):
    def __init__(self, callsign='LV'):
        super().__init__(callsign, 'SPACE')
        self.launch_phase = 0
        self.state.geospatial.alt_m = 0.0
    def step(self, dt): self.state.step_launch(self, dt)

class Falcon9(LaunchVehicle):
    def __init__(self): super().__init__('FALCON9'); self.mission = 'ISS_REFUEL'
class FalconHeavy(LaunchVehicle): pass
class BoeingSpaceliner(LaunchVehicle): pass

class BrandDataLoader:
    BRAND_DATA: Dict[str, Dict[str, Any]] = {
        'F35': {'missile_count': 4, 'missile_type': 'AIM-260', 'radar_detection_m': 150000, 'rcs_m2': 0.001, 'radar_detection_coeff': 1.0},
        'F22': {'missile_count': 4, 'missile_type': 'AIM-120D', 'radar_detection_m': 140000, 'rcs_m2': 0.0001, 'radar_detection_coeff': 1.0}
    }
    @staticmethod
    def get(brand: str) -> Dict[str, Any]: return BrandDataLoader.BRAND_DATA.get(brand, {})

FLIGHTS = [
    {'type': 'F35', 'callsign': 'BO1',    'airport': 'KIAH', 'team': 'WEST', 'pilot_name': 'Maj. Bo',                                 **BrandDataLoader.get('F35')},
    {'type': 'F35', 'callsign': 'MACE2',  'airport': 'KIAH', 'team': 'WEST', 'pilot_name': 'Capt. Alex "Mace" Knight',               **BrandDataLoader.get('F35')},
    {'type': 'F22', 'callsign': 'VIPER1', 'airport': 'KSFO', 'team': 'EAST', 'pilot_name': 'Capt. Jay "Ghost" Lee',                  **BrandDataLoader.get('F22')},
    {'type': 'F22', 'callsign': 'VIPER2', 'airport': 'KSFO', 'team': 'EAST', 'pilot_name': 'Capt. Mia "Storm" Kim',                  **BrandDataLoader.get('F22')},
    {'type': 'KC46','callsign': 'TANK1',  'airport': 'KIAH', 'team': 'WEST'},
    {'type': 'B737','callsign': 'CIV1',   'airport': 'KIAH', 'team': 'CIV'}
]

WAR_GAMES_CONFIG = {'sim_dt': 0.02, 'render_fps': 45, 'flights': FLIGHTS}

def load_texture(path):
    try:
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        img = Image.open(path).convert("RGB")
        img_data = img.tobytes()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return tex_id
    except Exception as e:
        logging.error(f'Texture load failed: {e}')
        return 0

def draw_text(x, y, z, text):
    glRasterPos3f(x, y, z)
    for ch in text: glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, ord(ch))

class Simulation:
    def __init__(self, cfg, duration=3600.0, seed=1, view_only=False):
        self.cfg = cfg
        self.duration = duration
        self.view_only = view_only
        self.weather = RealisticWeather()
        self.flights: List[Aircraft] = []
        self.tankers: List[KC46] = []
        self.missiles = []
        self.bo_ac: Optional[Aircraft] = None
        self.window_width, self.window_height = 1200, 800
        self.cam_rad = 1.05
        self.BACK_OFFSET_M = 500.0
        self.TAKEOFF_ALT_THRESH = 300.0
        self.TAKEOFF_BACK_FACTOR = 0.25
        self.idle_last = time.time()
        self.quadric = None
        self.tex_id = None
        self.google_ok = ensure_texture(GOOGLE_MAP_TEXTURE, GOOGLE_MAP_URL)
        self.earth_ok = ensure_texture(EARTH_TEXTURE, EARTH_URLS)
        self.takeoff_schedule: Dict[str, float] = {}
        self.takeoff_cleared: Dict[str, bool] = {}
        self._init_flights(seed)

    def _init_flights(self, seed):
        random.seed(seed)
        for f_cfg in self.cfg.get('flights', []):
            airport_code = f_cfg.get('airport')
            a_lat, a_lon = AIRPORTS.get(airport_code, (0.0, 0.0))
            runway_hdg_deg = RUNWAYS.get(airport_code, [{'hdg': 0.0}])[0]['hdg']
            cls_map: Dict[str, Any] = {'F35': F35, 'F22': F22, 'KC46': KC46, 'B737': Boeing737Max}
            cls = cls_map.get(f_cfg['type'], Aircraft)
            ac: Aircraft = cls()
            ac.state.geospatial.lat, ac.state.geospatial.lon = a_lat, a_lon
            ac.state.geospatial.alt_m = 0.0
            ac.runway_hdg = runway_hdg_deg * RAD
            ac.state.orientation.psi = ac.runway_hdg
            ac.state.flight.speed = 0.0
            ac.callsign = f_cfg.get('callsign', ac.callsign)
            ac.team = f_cfg.get('team', ac.team)
            ac.pilot_name = f_cfg.get('pilot_name', "")
            ac.state.radar.detection_range_m = f_cfg.get('radar_detection_m', FIRE_RANGE)
            ac.state.radar.rcs_m2 = f_cfg.get('rcs_m2', 1.0)
            ac.state.radar.detection_coeff = f_cfg.get('radar_detection_coeff', 1.0)
            ac.missile_type = f_cfg.get('missile_type', 'GENERIC')
            missiles = f_cfg.get('missile_count', 0)
            if missiles: ac.load_missiles(missiles, ac.missile_type)
            if ac.callsign == 'BO1': self.bo_ac = ac
            if isinstance(ac, KC46): self.tankers.append(ac)
            self.flights.append(ac)
        self.takeoff_schedule = {ac.callsign: 5.0 + i * 3.0 for i, ac in enumerate(self.flights) if isinstance(ac, Fighter)}
        self.takeoff_cleared = {k: False for k in self.takeoff_schedule}
        UNIVERSE.validate_required_fields()

    def set_camera(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, self.window_width / self.window_height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        if self.bo_ac and not self.bo_ac.state.flight.destroyed:
            hdg = self.bo_ac.state.orientation.psi
            takeoff = (self.bo_ac.state.geospatial.alt_m < self.TAKEOFF_ALT_THRESH)
            offset = self.BACK_OFFSET_M * (self.TAKEOFF_BACK_FACTOR if takeoff else 1.0)
            dlat = -(offset / 111320.0) * math.sin(hdg)
            dlon = -(offset / (111320.0 * math.cos(math.radians(self.bo_ac.state.geospatial.lat)))) * math.cos(hdg)
            cam_lat = self.bo_ac.state.geospatial.lat + dlat
            cam_lon = self.bo_ac.state.geospatial.lon + dlon
            alt_add = max(0.0, self.bo_ac.state.geospatial.alt_m) / EARTH_RADIUS_M
            current_rad = (1.03 if takeoff else self.cam_rad) + alt_add
            cam_x, cam_y, cam_z = geo_to_xyz(cam_lat, cam_lon, current_rad)
            tgt_x, tgt_y, tgt_z = geo_to_xyz(self.bo_ac.state.geospatial.lat, self.bo_ac.state.geospatial.lon, 1.02 + alt_add)
            gluLookAt(cam_x, cam_y, cam_z, tgt_x, tgt_y, tgt_z, 0, 1, 0)
        else:
            gluLookAt(0, 2, 4, 0, 0, 0, 0, 1, 0)

    def draw_earth(self):
        if self.tex_id: glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, self.tex_id)
        else: glColor3f(0.2, 0.6, 1.0)
        glPushMatrix(); gluSphere(self.quadric, 1.0, 64, 64); glPopMatrix()
        if self.tex_id: glDisable(GL_TEXTURE_2D)

    def draw_overlay(self):
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); gluOrtho2D(0, self.window_width, 0, self.window_height)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glColor3f(1, 1, 1)
        info_top = [
            f'Sim {(UNIVERSE.timestamp + datetime.timedelta(seconds=UNIVERSE.simulation.elapsed)).strftime("%H:%M:%S")}',
            f'Elapsed {UNIVERSE.simulation.elapsed:.1f}s', f'Speed x{UNIVERSE.simulation.sim_speed:.1f}'
        ]
        for i, l in enumerate(info_top):
            glRasterPos2f(10, self.window_height - 15 - 15 * i)
            for ch in l: glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(ch))
        for cs, cleared in self.takeoff_cleared.items():
            if cleared:
                msg = f'FCC TAKEOFF CLEARED {cs}'
                c_idx = list(self.takeoff_cleared.keys()).index(cs)
                glRasterPos2f(10, self.window_height - 70 - 15 * c_idx)
                for ch in msg: glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(ch))
        if self.bo_ac and not self.bo_ac.state.flight.destroyed:
            speed_kmh = self.bo_ac.state.flight.speed * 3.6
            if self.bo_ac.state.flight.on_ground:
                taxi_kmh = min(speed_kmh, 10.0) if self.bo_ac.state.flight.throttle < 0.25 else 0.0
                ground_kmh = 0.0 if taxi_kmh else speed_kmh
                air_kmh = 0.0
            else:
                taxi_kmh = ground_kmh = 0.0
                air_kmh = speed_kmh
            pitch, roll, yaw = self.bo_ac.state.euler_deg()
            heading_deg = (math.degrees(self.bo_ac.state.orientation.psi) % 360)
            var = magnetic_variation(self.bo_ac.state.geospatial.lat, self.bo_ac.state.geospatial.lon)
            mag_heading = (heading_deg - var + 360) % 360
            alt_m = self.bo_ac.state.geospatial.alt_m
            alt_ft = alt_m * FT_PER_M
            fuel_pct = self.bo_ac.state.flight.fuel_mass / self.bo_ac.state.flight.fuel_capacity * 100.0
            vals = [
                f'{self.bo_ac.callsign} F35 ({self.bo_ac.pilot_name})',
                f'Taxi {taxi_kmh:.0f} km/h',
                f'Ground {ground_kmh:.0f} km/h',
                f'Air {air_kmh:.0f} km/h',
                f'Alt {alt_m:.0f} m ({alt_ft:.0f} ft)',
                f'GroundState {self.bo_ac.state.flight.on_ground}',
                f'Hdg {heading_deg:.1f}°',
                f'MagHdg {mag_heading:.1f}° ({var:+.1f}°)',
                f'Pitch {pitch:.1f}°',
                f'Roll {roll:.1f}°',
                f'Yaw {yaw:.1f}°',
                f'Lat {self.bo_ac.state.geospatial.lat:.4f}',
                f'Lon {self.bo_ac.state.geospatial.lon:.4f}',
                f'Thrust {self.bo_ac.state.flight.throttle * 100:.0f}%',
                f'Accel {self.bo_ac.state.physical.acceleration:.1f} m/s²',
                f'Fuel {self.bo_ac.state.flight.fuel_mass:.0f} kg ({fuel_pct:.0f}%)',
                f'Missiles {self.bo_ac.state.flight.missile_ready}/{self.bo_ac.state.flight.missile_total}'
            ]
            if self.bo_ac.v1_time is not None: vals.append(f'V1 {self.bo_ac.v1_time:.1f}s')
            if self.bo_ac.v2_time is not None: vals.append(f'V2 {self.bo_ac.v2_time:.1f}s')
            vals += [
                'Pilot Input:',
                f'  Throttle {self.bo_ac.pilot_input.throttle * 100:.0f}%',
                f'  Yoke {self.bo_ac.pilot_input.yoke:.2f}',
                f'  Roll {self.bo_ac.pilot_input.roll:.2f}',
                f'  Rudder {self.bo_ac.pilot_input.rudder:.2f}',
                f'  Trim {self.bo_ac.pilot_input.trim:.2f}',
                f'  Flap {self.bo_ac.pilot_input.flap_cmd}',
                f'  Gear {"DOWN" if self.bo_ac.pilot_input.gear_cmd else "UP"}'
            ]
            for idx, l in enumerate(vals):
                glRasterPos2f(10, self.window_height - 200 - 15 * idx)
                for ch in l: glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(ch))
            for idx, msl in enumerate(self.bo_ac.missiles):
                glRasterPos2f(10, self.window_height - 200 - 15 * (len(vals) + idx))
                label = f'M{idx + 1}: {msl.status}'
                for ch in label: glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(ch))
        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)

    def draw_flights(self):
        for ac in self.flights:
            if ac.state.flight.destroyed: continue
            alt_factor = 1.02 + max(0.0, ac.state.geospatial.alt_m) / EARTH_RADIUS_M
            glPointSize(8.0 if (ac.callsign == 'BO1' and ac.takeoff_authorized) else 6.0)
            glBegin(GL_POINTS)
            if ac.team == 'WEST': glColor3f(0.0, 0.0, 1.0)
            elif ac.team == 'SPACE': glColor3f(0.0, 1.0, 1.0)
            elif ac.team == 'CIV': glColor3f(1.0, 1.0, 0.0)
            else: glColor3f(1.0, 0.0, 0.0)
            if ac.callsign == 'BO1' and ac.takeoff_authorized:
                flash = 0.5 + 0.5 * math.sin(UNIVERSE.simulation.elapsed * 10.0)
                glColor3f(0.0, flash, 0.0)
            x, y, z = geo_to_xyz(ac.state.geospatial.lat, ac.state.geospatial.lon, alt_factor)
            glVertex3f(x, y, z)
            glEnd()
            glColor3f(1.0, 1.0, 1.0)
            x2, y2, z2 = geo_to_xyz(ac.state.geospatial.lat, ac.state.geospatial.lon, alt_factor + 0.02)
            draw_text(x2, y2, z2, f'{ac.callsign}')

    def draw_missiles(self):
        if not self.missiles: return
        glPointSize(4.0); glColor3f(1.0, 1.0, 0.0); glBegin(GL_POINTS)
        for m in self.missiles:
            if m.get('terminated'): continue
            alt_factor = 1.02 + max(0.0, m['shooter'].state.geospatial.alt_m) / EARTH_RADIUS_M
            x, y, z = geo_to_xyz(m['lat'], m['lon'], alt_factor)
            glVertex3f(x, y, z)
        glEnd()

    def sim_step(self, dt_real):
        if dt_real <= 0: return
        steps = max(1, int(round(dt_real * UNIVERSE.simulation.sim_speed / self.cfg.get('sim_dt', 0.02))))
        for _ in range(steps):
            UNIVERSE.simulation.elapsed += self.cfg.get('sim_dt', 0.02)
            for cs, t_auth in self.takeoff_schedule.items():
                if not self.takeoff_cleared[cs] and UNIVERSE.simulation.elapsed >= t_auth:
                    self.takeoff_cleared[cs] = True
                    for ac in self.flights:
                        if ac.callsign == cs:
                            ac.takeoff_authorized = True
                            logging.info(f'FCC TAKEOFF AUTHORIZATION for {cs} at t={UNIVERSE.simulation.elapsed:.2f}s')
                            break
            for ac in self.flights:
                if ac.state.flight.destroyed or isinstance(ac, OrbitalAsset): continue
                tgt_alt, tgt_hdg, th = ac.pilot.decide(ac, self.flights, self.missiles, self.tankers)
                ac.ap.set_targets(tgt_alt, tgt_hdg)
                ac.state.flight.throttle = min(1.0, max(0.0, th))
                ac.pilot_input.throttle = ac.state.flight.throttle
            for ac in self.flights:
                if ac.state.flight.destroyed: continue
                ac.step(self.cfg.get('sim_dt', 0.02))
            for shooter in self.flights:
                if shooter.state.flight.destroyed or shooter.state.flight.on_ground or shooter.state.flight.speed < 200.0 or shooter.state.flight.missile_ready == 0 or not shooter.state.radar.active: continue
                for target in self.flights:
                    if target is shooter or target.team == shooter.team or not isinstance(target, Fighter) or target.state.flight.destroyed: continue
                    bearing = calc_heading(shooter.state.geospatial.lat, shooter.state.geospatial.lon, target.state.geospatial.lat, target.state.geospatial.lon)
                    angle_off = abs(((bearing - shooter.state.orientation.psi + math.pi) % (2 * math.pi)) - math.pi)
                    if angle_off > shooter.state.radar.cone_deg * RAD * 0.5: continue
                    rng = haversine(shooter.state.geospatial.lat, shooter.state.geospatial.lon, target.state.geospatial.lat, target.state.geospatial.lon) * 1000.0
                    eff_det_range = shooter.state.radar.detection_coeff * shooter.state.radar.detection_range_m * (max(target.state.radar.rcs_m2, 1e-10) ** 0.25)
                    if rng <= FIRE_RANGE and rng <= eff_det_range:
                        msl_obj = shooter.fire_missile()
                        if not msl_obj: break
                        self.missiles.append({'obj': msl_obj, 'shooter': shooter, 'target': target, 'dist': rng, 'init_dist': rng if rng > 0 else 1.0, 'lat': shooter.state.geospatial.lat, 'lon': shooter.state.geospatial.lon, 'terminated': False})
                        logging.debug(f'{shooter.callsign} FIRE {msl_obj.missile_type} -> {target.callsign} RCS_lock {target.state.radar.rcs_m2:.6f} rng {rng:.0f} eff_rng {eff_det_range:.0f} angle {math.degrees(angle_off):.1f}')
                        if msl_obj.missile_type == 'AIM-260':
                            logging.debug(f'{msl_obj.name} AMRAM260_AI INIT engine_fuse=True ctrl_state={{"phase":"boost"}} dl_link=True')
                        break
            for m in self.missiles:
                if m.get('terminated'): continue
                m['obj'].step(self.cfg.get('sim_dt', 0.02))
                m['dist'] -= m['obj'].speed * self.cfg.get('sim_dt', 0.02)
                frac = 1.0 if m['init_dist'] <= 0 else max(0, min(1, (m['init_dist'] - m['dist']) / m['init_dist']))
                m['lat'] = m['shooter'].state.geospatial.lat + (m['target'].state.geospatial.lat - m['shooter'].state.geospatial.lat) * frac
                m['lon'] = m['shooter'].state.geospatial.lon + (m['target'].state.geospatial.lon - m['shooter'].state.geospatial.lon) * frac
                ir_sig = m['target'].state.material.temperature + 273.15
                dlq = max(0.0, 1.0 - m['dist'] / m['init_dist'])
                logging.debug(f'{m["obj"].name} AI_CTRL dist {m["dist"]:.1f}m frac {frac:.3f} dlq {dlq:.3f} IR {ir_sig:.1f}K tgt_RCS {m["target"].state.radar.rcs_m2:.6f}')
                if m['obj'].missile_type == 'AIM-260':
                    logging.debug(f'{m["obj"].name} AMRAM260_AI STATUS dist {m["dist"]:.1f}m engine_on {m["dist"] > 2000:.0f} dlq {dlq:.3f} IR {ir_sig:.1f}K tgt_RCS {m["target"].state.radar.rcs_m2:.6f}')
                if m['dist'] <= 0:
                    if not m['target'].state.flight.destroyed:
                        logging.info(f'{m["target"].callsign} destroyed')
                        m['target'].state.flight.destroyed = True
                    m['obj'].status = 'EXPENDED'
                    m['terminated'] = True
            if UNIVERSE.simulation.elapsed >= self.duration:
                logging.info('Simulation End')
                glutLeaveMainLoop()
                return

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.set_camera()
        self.draw_earth()
        self.draw_flights()
        self.draw_missiles()
        self.draw_overlay()
        glutSwapBuffers()

    def idle(self):
        current = time.time()
        dt_real = current - self.idle_last
        self.idle_last = current
        if not self.view_only: self.sim_step(dt_real)
        glutPostRedisplay()

    def keyboard(self, key, _x, _y):
        if key == b'\x1b': sys.exit(0)
        if key == b'+': UNIVERSE.simulation.sim_speed = min(10.0, UNIVERSE.simulation.sim_speed + 0.1)
        if key == b'-': UNIVERSE.simulation.sim_speed = max(0.1, UNIVERSE.simulation.sim_speed - 0.1)

    def run(self):
        try:
            glutInit(sys.argv)
            glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
            glutInitWindowSize(self.window_width, self.window_height)
            glutCreateWindow(b"Bo War Game Sim")
            glEnable(GL_DEPTH_TEST); glClearColor(0.05, 0.05, 0.1, 1.0)
            self.quadric = gluNewQuadric(); gluQuadricTexture(self.quadric, GL_TRUE)
            base_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
            tex_choice = GOOGLE_MAP_TEXTURE if self.google_ok else EARTH_TEXTURE
            self.tex_id = load_texture(os.path.join(base_dir, tex_choice))
            glutDisplayFunc(self.display)
            glutIdleFunc(self.idle)
            glutKeyboardFunc(self.keyboard)
            glutMainLoop()
        except Exception as e:
            logging.error(f'OpenGL init failed: {e}')

class Task(Enum): anim = auto()

def run_war_games(dt=0.02, duration=3600.0, seed=1, view_only=False):
    cfg = WAR_GAMES_CONFIG
    cfg['sim_dt'] = dt
    sim = Simulation(cfg, duration=duration, seed=seed, view_only=view_only)
    sim.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--war_games", action="store_true")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--duration", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render_fps", type=int, default=45)
    parser.add_argument("--view_only", action="store_true")
    args = parser.parse_args()
    if args.war_games:
        WAR_GAMES_CONFIG['sim_dt'] = args.dt
        sim = Simulation(WAR_GAMES_CONFIG, duration=args.duration, seed=args.seed, view_only=args.view_only)
        sim.run()
        sys.exit(0)
